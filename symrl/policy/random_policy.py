try:
    from .base_policy import BasePolicy
except ImportError:
    from base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, observation):
        return self.action_space.sample()