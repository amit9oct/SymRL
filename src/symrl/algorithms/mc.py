import sys
root_dir = __file__.split('symrl')[0]
if root_dir not in sys.path:
    sys.path.append(root_dir)
from symrl.func_approximator.base_approx import BaseFuncApproximator
from symrl.algorithms.base_aglo import BaseAlgo

class MonteCarlo(BaseAlgo):
    def __init__(self, num_actions : int, func_approximator : BaseFuncApproximator, gamma=0.99):
        super().__init__(func_approximator)
        self.gamma = gamma
        self.num_actions = num_actions
        self.episode = []

    def update(self, state, action, reward, next_state, done):
        self.episode.append((state, action, reward))
        if done:
            self._update_q_values()
            self.episode = []
    
    def _update_q_values(self):
        G = 0
        for t in range(len(self.episode)-1, -1, -1):
            state, action, reward = self.episode[t]
            G = self.gamma * G + reward
            self.func_approximator.update_q(state, action, G)