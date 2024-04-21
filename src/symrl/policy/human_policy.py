try:
    from .base_policy import BasePolicy
except ImportError:
    from base_policy import BasePolicy


class HumanPolicy(BasePolicy):
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, observation):
        print(f"Enter the action for the given equation: {observation}")
        return input()
