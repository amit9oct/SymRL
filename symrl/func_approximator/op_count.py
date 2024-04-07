from symrl.environment.sympy_env import SympyEnv
from symrl.func_approximator.base_approx import FeatureExtractor
import numpy as np


class OpCountFeatureExtractor(FeatureExtractor):
    def __init__(self, env: SympyEnv):
        self.env = env

    def __call__(self, observation):
        lhs, rhs = SympyEnv.get_lhs_rhs_op_count(observation)
        return np.array([lhs, rhs])

    def pretty_print_feature_extractor(self) -> str:
        feature_names = ["op_count_lhs", "op_count_rhs"]
        return str(feature_names)

    def pretty_print_state(self, state) -> str:
        state_str = str(self(state))
        return state_str

    def pretty_print_action(self, action) -> str:
        action_str = self.env.action_space.actions[action]
        return action_str
    
    @property
    def num_features(self):
        return 2