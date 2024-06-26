import numpy as np
try:
    from .base_policy import BasePolicy
except ImportError:
    from base_policy import BasePolicy
import sys
import json
root_dir = __file__.split('symrl')[0]
if root_dir not in sys.path:
    sys.path.append(root_dir)
from symrl.func_approximator.base_approx import BaseFuncApproximator

class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, epsilon, num_actions, func_approximator: BaseFuncApproximator, break_ties_randomly=True):
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.func_approximator = func_approximator
        self.break_ties_randomly = break_ties_randomly

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            # With probability epsilon, select a random action (explore)
            return np.random.randint(self.num_actions)
        else:
            # Otherwise, select the action with the highest predicted Q-value (exploit)
            q_values = [self.func_approximator.predict_q(state, action) for action in range(self.num_actions)]
            if self.break_ties_randomly:
                max_q_value = np.max(q_values)
                return np.random.choice(np.where(q_values == max_q_value)[0])
            else:
                return np.argmax(q_values)

    def pretty_print_state(self, state):
        state_approx = self.func_approximator.pretty_print_state(state)
        return state_approx
    
    def pretty_print_action(self, action):
        action_approx = self.func_approximator.pretty_print_action(action)
        return action_approx
    
    def pretty_print_policy(self):
        approx_policy = self.func_approximator.pretty_print_approximator()
        approx_policy += f"\nEpsilon: {self.epsilon}\nGeedy {'randomly' if self.break_ties_randomly else 'deterministically'} breaks ties."
        return approx_policy

    def save(self, folder):
        filename = f"{folder}/EpsilonGreedyPolicy.json"
        with open(filename, 'w') as f:
            json.dump({"epsilon": self.epsilon, "num_actions": self.num_actions, "break_ties_randomly": self.break_ties_randomly}, f)
        self.func_approximator.save(folder)
    
    def load(folder):
        filename = f"{folder}/EpsilonGreedyPolicy.json"
        with open(filename, 'r') as f:
            settings = json.load(f)
        func_approximator = BaseFuncApproximator.load(folder)
        return EpsilonGreedyPolicy(settings["epsilon"], settings["num_actions"], func_approximator, settings["break_ties_randomly"])
    
    def __str__(self) -> str:
        return f"EpsilonGreedy(eps={self.epsilon})"