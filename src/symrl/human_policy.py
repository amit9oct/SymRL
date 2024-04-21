from policy.base_policy import BasePolicy
from func_approximator.base_approx import BaseFuncApproximator

class HumanPolicy(BasePolicy):
    def __init__(self, num_actions, func_approximator: BaseFuncApproximator, break_ties_randomly=True):
        self.num_actions = num_actions
        self.func_approximator = func_approximator
        self.num_actions = num_actions
        self.break_ties_randomly = break_ties_randomly
        self._call_count = 0
    
    def reset(self):
        self._call_count = 0

    def select_action(self, observation):
        # These are the actions ["CLV", "MVR", "MVL", "DIV", "SIM", "CLC", "MCR", "MCL"]
        assert self._call_count < self.num_actions, f"Number of actions exceeded: {self._call_count}"
        if self._call_count == 1 or self._call_count == 7:
            self._call_count += 1 # Skip MVL as we will never want to move variables to LHS
        action = self._call_count % self.num_actions
        self._call_count += 1 # Keep looping through the actions
        self._call_count %= self.num_actions
        return action

    def pretty_print_state(self, state):
        state_approx = self.func_approximator.pretty_print_state(state)
        return state_approx
    
    def pretty_print_action(self, action):
        action_approx = self.func_approximator.pretty_print_action(action)
        return action_approx
    
    def pretty_print_policy(self):
        approx_policy = self.func_approximator.pretty_print_approximator()
        approx_policy += f"\nGeedy {'randomly' if self.break_ties_randomly else 'deterministically'} breaks ties."
        return approx_policy

    def save(self, filename):
        pass

    def __str__(self) -> str:
        return "Human Policy"