import numpy as np
try:
    from .base_policy import BasePolicy
except ImportError:
    from base_policy import BasePolicy
import sys
root_dir = __file__.split('symrl')[0]
if root_dir not in sys.path:
    sys.path.append(root_dir)
from symrl.func_approximator.base_approx import BaseFuncApproximator

class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, epsilon, num_actions, func_approximator: BaseFuncApproximator):
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.func_approximator = func_approximator

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            # With probability epsilon, select a random action (explore)
            return np.random.randint(self.num_actions)
        else:
            # Otherwise, select the action with the highest predicted Q-value (exploit)
            q_values = [self.func_approximator.predict_q(state, action) for action in range(self.num_actions)]
            return np.argmax(q_values)
