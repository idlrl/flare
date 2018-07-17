from flare.framework.computation_task import ComputationTask
from flare.algorithm_zoo.simple_algorithms import SimpleAC, SimpleQ, SimpleSARSA
from flare.model_zoo.simple_models import SimpleModelAC, SimpleModelQ, GaussianPolicyModel
import numpy as np
import torch.nn as nn
import unittest
import math
import gym


def unpack_exps(exps):
    return [np.array(l).astype('int' if i==2 else 'float32') \
            for i, l in enumerate(zip(*exps))]


def sample(past_exps, n):
    sampled = []
    while len(sampled) < n:
        idx = np.random.randint(0, len(past_exps) - 1)
        if past_exps[idx][3][0]:  ## episode end sampled
            continue
        sampled.append((
            past_exps[idx][0],  ## ob
            past_exps[idx + 1][0],  ## next_ob
            past_exps[idx][1],
            past_exps[idx + 1][1],
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
        on_policies = [False, True, True]
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
                if discrete_action:
                    alg = SimpleSARSA(model=q_model, epsilon=0.1)
                else:
                    alg = SimpleAC(model=GaussianPolicyModel(
                        dims=state_shape,
                        action_dims=num_actions,
                        mlp=mlp,
                        std=1.0))
            else:
                alg = SimpleQ(
                    model=q_model,
                    exploration_end_steps=500000,
                    update_ref_interval=100)

            print "algorithm: " + alg.__class__.__name__

            ct = ComputationTask(algorithm=alg, hyperparas=dict(lr=1e-4))
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
                game_over = False
                for t in range(max_steps):
                    res, _ = ct.predict(inputs=dict(sensor=np.array(
                        [ob]).astype("float32")))

                    ## when discrete_action is True, this is a scalar
                    ## otherwise it's a floating vector
                    pred_action = res["action"][0]

                    ## end before the env wrongly gives game_over=True for a timeout case
                    if t == max_steps - 1 or game_over:
                        past_exps.append((ob, pred_action, [0], [game_over]))
                        break
                    else:
                        next_ob, reward, next_is_over, _ = env.step(
                            pred_action[0] if discrete_action else pred_action)
                        reward /= 100
                        episode_reward += reward
                        past_exps.append(
                            (ob, pred_action, [reward], [game_over]))

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
                        sensor, next_sensor, action, next_action, reward, next_episode_end \
                            = unpack_exps(exps)
                        cost = ct.learn(
                            inputs=dict(sensor=sensor),
                            next_inputs=dict(sensor=next_sensor),
                            next_episode_end=dict(
                                next_episode_end=next_episode_end),
                            actions=dict(action=action),
                            next_actions=dict(action=next_action),
                            rewards=dict(reward=reward))
                        ## we clear the exp buffer for on-policy
                        if on_policy:
                            past_exps = []

                    ob = next_ob
                    game_over = next_is_over

                if n % 50 == 0:
                    print("episode reward: %f" % episode_reward)

                average_episode_reward.append(episode_reward)
                if len(average_episode_reward) > 20:
                    average_episode_reward.pop(0)

                ### once hit the threshold, we don't bother running
                if sum(average_episode_reward) / len(
                        average_episode_reward) > threshold:
                    print "Test terminates early due to threshold satisfied!"
                    break

            ### compuare the average episode reward to reduce variance
            self.assertGreater(
                sum(average_episode_reward) / len(average_episode_reward),
                threshold)


if __name__ == "__main__":
    unittest.main()
