try:
    from .base_policy import BasePolicy
except ImportError:
    from base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, observation):
        return self.action_space.sample()
    
    def pretty_print_state(self, state):
        return str(state)
    
    def pretty_print_action(self, action):
        return str(action)
    
    def pretty_print_policy(self):
        return f"Random Policy {self.action_space.actions}"

    def save(self, filename):
        pass

    def __str__(self) -> str:
        return "Random Policy"