import sys
root_dir = __file__.split('symrl')[0]
if root_dir not in sys.path:
    sys.path.append(root_dir)
from symrl.func_approximator.base_approx import BaseFuncApproximator
from symrl.algorithms.base_aglo import BaseAlgo

class TDZero(BaseAlgo):
    def __init__(self, func_approximator : BaseFuncApproximator, gamma=0.99):
        super().__init__(func_approximator)
        self.gamma = gamma

    def update(self, state, action, reward, next_state, done):
        current_value = self.func_approximator.predict_q(state, action)
        next_value = 0 if done else max(self.func_approximator.predict_q(next_state, a) for a in range(self.func_approximator.num_actions))
        target = reward + self.gamma * next_value
        self.func_approximator.update_q(state, action, target)
