from flare.framework.computation_task import ComputationTask
from flare.algorithm_zoo.simple_algorithms import SimpleAC, SimpleQ, SimpleSARSA, OffPolicyAC
from flare.model_zoo.simple_models import SimpleModelAC, SimpleModelQ, GaussianPolicyModel
from flare.algorithm_zoo.successor_representation import SuccessorRepresentationQ
from flare.model_zoo.successor_representation_models import SimpleSRModel
import numpy as np
import torch.nn as nn
import unittest
import math
import gym
import glog


def unpack_exps(exps):
    ret = []
    for i, l in enumerate(zip(*exps)):
        dct = dict()
        for k in l[0].keys():
            dct[k] = np.vstack([e[k] for e in l])
        ret.append(dct)
    return ret


def sample(past_exps, n):
    def is_episode_end(alive):
        return alive.values()[0][0][0] <= 0  # timeout or die

    sampled = []
    while len(sampled) < n:
        idx = np.random.randint(0, len(past_exps) - 1)
        if is_episode_end(past_exps[idx][3]):  ## episode end sampled
            continue
        sampled.append((
            past_exps[idx][0],  ## inputs
            past_exps[idx + 1][0],  ## next_inputs
            past_exps[idx][1],  ## actions
            past_exps[idx + 1][1],  ## next_actions
            past_exps[idx][2],
            past_exps[idx + 1][3]))
    return sampled


class TestGymGame(unittest.TestCase):
    def test_gym_games(self):
        """
        Test games in OpenAI gym.
        """

        games = ["MountainCar-v0", "CartPole-v0", "Pendulum-v0"]
        final_rewards_thresholds = [
            -1.5,  ## drive to the right top in 150 steps (timeout is -2.0)
            1.5,  ## hold the pole for at least 150 steps
            -3.0  ## can swing the stick to the top most of the times
        ]
        on_policies = [False, True, False]
        discrete_actions = [True, True, False]

        for game, threshold, on_policy, discrete_action in \
            zip(games, final_rewards_thresholds, on_policies, discrete_actions):

            env = gym.make(game)
            state_shape = env.observation_space.shape[0]
            if discrete_action:
                num_actions = env.action_space.n
            else:
                num_actions = env.action_space.shape[0]

            hidden_size = 256

            mlp = nn.Sequential(
                nn.Linear(state_shape, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU())

            q_model = SimpleModelQ(
                dims=state_shape,
                num_actions=num_actions,
                mlp=nn.Sequential(mlp, nn.Linear(hidden_size, num_actions)))

            if on_policy:
                alg = SimpleSARSA(model=q_model, epsilon=0.1)
                # alg = SuccessorRepresentationQ(
                #     ## much slower than SARSA because of more things to learn
                #     model=SimpleSRModel(
                #         dims=state_shape,
                #         hidden_size=hidden_size,
                #         num_actions=num_actions, ),
                #     exploration_end_steps=20000)
            else:
                if discrete_action:
                    alg = SimpleQ(
                        model=q_model,
                        exploration_end_steps=200000,
                        update_ref_interval=100)
                else:
                    alg = OffPolicyAC(
                        model=GaussianPolicyModel(
                            dims=state_shape,
                            action_dims=num_actions,
                            mlp=mlp,
                            std=1.0),
                        epsilon=0.2)

            glog.info("algorithm: " + alg.__class__.__name__)

            ct = ComputationTask("RL", algorithm=alg, hyperparas=dict(lr=1e-4))
            batch_size = 32
            if not on_policy:
                train_every_steps = batch_size / 4
                buffer_size_limit = 200000

            max_episode = 10000

            average_episode_reward = []
            past_exps = []
            max_steps = env._max_episode_steps
            for n in range(max_episode):
                ob = env.reset()
                episode_reward = 0
                alive = 1
                for t in range(max_steps):
                    inputs = dict(sensor=np.array([ob]).astype("float32"))
                    res, _ = ct.predict(inputs=inputs)

                    ## when discrete_action is True, this is a scalar
                    ## otherwise it's a floating vector
                    pred_action = res["action"][0]

                    ## end before the env wrongly gives game_over=True for a timeout case
                    if t == max_steps - 1:
                        past_exps.append(
                            (inputs, res, dict(reward=[[0]]),
                             dict(alive=[[-1]])))  ## -1 denotes timeout
                        break
                    elif (not alive):
                        past_exps.append((inputs, res, dict(reward=[[0]]),
                                          dict(alive=[[alive]])))
                        break
                    else:
                        next_ob, reward, next_is_over, _ = env.step(
                            pred_action[0] if discrete_action else pred_action)
                        reward /= 100
                        episode_reward += reward
                        past_exps.append((inputs, res, dict(reward=[[reward]]),
                                          dict(alive=[[alive]])))

                    ## only for off-policy training we use a circular buffer
                    if (not on_policy) and len(past_exps) > buffer_size_limit:
                        past_exps.pop(0)

                    ## compute the learning condition
                    learn_cond = False
                    if on_policy:
                        learn_cond = (len(past_exps) >= batch_size)
                    else:
                        learn_cond = (
                            t % train_every_steps == train_every_steps - 1)

                    if learn_cond:
                        exps = sample(past_exps, batch_size)
                        sampled_inputs, next_sampled_inputs, sampled_actions, \
                            next_sampled_actions, reward, next_alive = unpack_exps(exps)
                        cost = ct.learn(
                            inputs=sampled_inputs,
                            next_inputs=next_sampled_inputs,
                            next_alive=next_alive,
                            actions=sampled_actions,
                            next_actions=next_sampled_actions,
                            rewards=reward)
                        ## we clear the exp buffer for on-policy
                        if on_policy:
                            past_exps = []

                    ob = next_ob
                    ### bool must be converted to int for correct computation
                    alive = 1 - int(next_is_over)

                if n % 50 == 0:
                    glog.info("episode reward: %f" % episode_reward)

                average_episode_reward.append(episode_reward)
                if len(average_episode_reward) > 20:
                    average_episode_reward.pop(0)

                ### once hit the threshold, we don't bother running
                if sum(average_episode_reward) / len(
                        average_episode_reward) > threshold:
                    glog.info(
                        "Test terminates early due to threshold satisfied!")
                    break

            ### compuare the average episode reward to reduce variance
            self.assertGreater(
                sum(average_episode_reward) / len(average_episode_reward),
                threshold)


if __name__ == "__main__":
    unittest.main()
