from symrl.environment.sympy_env import SympyEnv
from symrl.func_approximator.base_approx import FeatureExtractor
import numpy as np


class RelTermVarConstCountFeatureExtractor(FeatureExtractor):
    def __init__(self, env: SympyEnv):
        self.env = env

    def __call__(self, observation):
        all_lhs_var_count, all_rhs_var_count = SympyEnv.get_lhs_rhs_var_count(observation)
        all_lhs_const_count, all_rhs_const_count = SympyEnv.get_lhs_rhs_const_count(observation)
        all_lhs_term_count, all_rhs_term_count = SympyEnv.get_lhs_rhs_term_count(observation)
        all_const_count = all_lhs_const_count + all_rhs_const_count
        all_var_count = all_lhs_var_count + all_rhs_var_count
        all_term_count = all_lhs_term_count + all_rhs_term_count
        return np.array([
            all_lhs_var_count / all_var_count,
            all_rhs_var_count / all_var_count, 
            all_lhs_const_count / all_const_count, 
            all_rhs_const_count / all_const_count,
            all_lhs_term_count / all_term_count,
            all_rhs_term_count / all_term_count])

    def pretty_print_feature_extractor(self) -> str:
        feature_names = [
            "rel_lhs_var_count",
            "rel_rhs_var_count",
            "rel_lhs_const_count",
            "rel_rhs_const_count",
            "rel_lhs_term_count",
            "rel_rhs_term_count"
        ]
        return str(feature_names)

    def pretty_print_state(self, state) -> str:
        state_str = str(self(state))
        return state_str

    def pretty_print_action(self, action) -> str:
        action_str = self.env.action_space.actions[action]
        return action_str
    
    @property
    def num_features(self):
        return 6