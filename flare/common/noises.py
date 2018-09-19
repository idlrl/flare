import numpy as np


class GaussianNoise(object):
    def __init__(self, action_dims, mu=0, sigma=0.1):
        self.action_dims = action_dims
        self.mu = mu
        self.sigma = sigma

    def reset(self):
        pass

    def noise(self):
        return np.random.normal(size=self.action_dims) * self.sigma + self.mu


# Ornstein-Uhlenbeck process: 
# https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
class OUNoise(object):
    def __init__(self, action_dims, mu=0, sigma=0.2, theta=0.15, dt=1e-2):
        self.action_dims = action_dims
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dims) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=len(x))
        self.state = x + dx
        return self.state
