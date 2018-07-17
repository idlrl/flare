from flare.framework.computation_task import ComputationTask
from flare.algorithm_zoo.simple_algorithms import SimpleQ
from flare.model_zoo.simple_models import SimpleRNNModelQ
import numpy as np
import torch.nn as nn
import unittest
import math
import gym


def unpack_exps_seqs(exps_seqs):
    return zip(*[[np.array(l) for l in zip(*exps)] \
                 for exps in exps_seqs])


def sample(past_exps, n, n_seqs):
    """
    We randomly sample n_seqs sequence whose total length is up to n
    """
    ret = []
    for i in range(n_seqs):
        start = np.random.randint(0, len(past_exps) - 1)
        indices = []
        while start < len(past_exps) \
              and len(indices) < n / n_seqs:
            indices.append(start)
            if not past_exps[start][4]:
                break
            start += 1
        ret.append([past_exps[i] for i in indices])
    return ret


class TestGymGame(unittest.TestCase):
    def test_gym_games(self):
        """
        Test games in OpenAI gym.
        """

        games = ["MountainCar-v0", "CartPole-v0"]
        final_rewards_thresholds = [
            -1.8,  ## drive to the right top in 180 steps (timeout is -2.0)
            1.5  ## hold the pole for at least 150 steps
        ]
        exploration_steps = [
            500000,
            50000,
        ]

        for game, threshold, esteps in zip(games, final_rewards_thresholds,
                                           exploration_steps):
            env = gym.make(game)
            state_shape = env.observation_space.shape[0]
            num_actions = env.action_space.n
            batch_size = 32
            num_seqs = batch_size / 2
            train_every_steps = batch_size / 8
            buffer_size_limit = 100000
            max_episode = 5000
            hidden_size = 128

            mlp = nn.Sequential(
                nn.Linear(state_shape, hidden_size),
                nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU())

            alg = SimpleQ(
                model=SimpleRNNModelQ(
                    dims=state_shape, num_actions=num_actions, mlp=mlp),
                exploration_end_steps=esteps,
                update_ref_interval=100 * batch_size / num_seqs, )

            print "algorithm: " + alg.__class__.__name__

            ct = ComputationTask(algorithm=alg, hyperparas=dict(lr=1e-4))

            average_episode_reward = []
            past_exps = []
            max_steps = env._max_episode_steps
            for n in range(max_episode):
                ob = env.reset()
                episode_reward = 0
                state = [0] * hidden_size  ## all zeros at episode start
                for t in range(max_steps):
                    res, states = ct.predict(
                        inputs=dict(sensor=np.array([ob]).astype("float32")),
                        states=dict(state=np.array([state]).astype("float32")))
                    pred_action = res["action"][0][0]

                    next_ob, reward, next_is_over, _ = env.step(pred_action)
                    reward /= 100
                    episode_reward += reward

                    past_exps.append(
                        (ob, next_ob, [pred_action], [reward],
                         [not next_is_over], state, states["state"][0]))

                    ### update the state
                    state = states["state"][0]

                    if len(past_exps) > buffer_size_limit:
                        past_exps.pop(0)

                    ## compute the learning condition
                    if t % train_every_steps == train_every_steps - 1:
                        exps = sample(past_exps, batch_size,
                                      num_seqs)  ## sample some exps
                        sensor, next_sensor, action, reward, next_episode_end, \
                            states, next_states = unpack_exps_seqs(exps)
                        ## we only take the first entries for states and next_states
                        states = [st[0] for st in states]
                        next_states = [st[0] for st in next_states]
                        cost = ct.learn(
                            inputs=dict(sensor=list(
                                sensor)),  ## one more level for seq
                            states=dict(state=states),  ## one less level
                            next_inputs=dict(sensor=list(
                                next_sensor)),  ## one more level for seq
                            next_states=dict(state=next_states),
                            next_episode_end=dict(next_episode_end=list(
                                next_episode_end)),  ## one more level for seq
                            actions=dict(action=list(action)),
                            rewards=dict(reward=list(reward)))

                    ob = next_ob

                    ## end before the Gym wrongly gives game_over=True for a timeout case
                    if t == max_steps - 2 or next_is_over:
                        break

                if n % 50 == 0:
                    print("episode reward: %f" % episode_reward)

                average_episode_reward.append(episode_reward)
                if len(average_episode_reward) > 20:
                    average_episode_reward.pop(0)

                ### once hit the threshold, we don't bother running
                if sum(average_episode_reward) / len(
                        average_episode_reward) > threshold:
                    break

            ### compuare the average episode reward to reduce variance
            self.assertGreater(
                sum(average_episode_reward) / len(average_episode_reward),
                threshold)


if __name__ == "__main__":
    unittest.main()
