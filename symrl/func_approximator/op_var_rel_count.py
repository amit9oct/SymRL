from symrl.environment.sympy_env import SympyEnv
from symrl.func_approximator.base_approx import FeatureExtractor
import numpy as np


class OpVarRelCountFeatureExtractor(FeatureExtractor):
    def __init__(self, env: SympyEnv):
        self.env = env

    def __call__(self, observation):
        lhs_term_cnt, rhs_term_cnt = SympyEnv.get_lhs_rhs_term_count(observation)
        all_count = lhs_term_cnt + rhs_term_cnt
        rel_lhs_term_count = lhs_term_cnt / all_count if all_count != 0 else 0
        rel_rhs_term_count = rhs_term_cnt / all_count if all_count != 0 else 0
        lhs_var_count, rhs_var_count = SympyEnv.get_lhs_rhs_var_count(observation)
        all_var_count = lhs_var_count + rhs_var_count
        rel_lhs_var_count = lhs_var_count / all_var_count if all_var_count != 0 else 0
        rel_rhs_var_count = rhs_var_count / all_var_count if all_var_count != 0 else 0
        return np.array([rel_lhs_term_count, rel_rhs_term_count, rel_lhs_var_count, rel_rhs_var_count])

    def pretty_print_feature_extractor(self) -> str:
        feature_names = [
            "rel_lhs_term_count",
            "rel_rhs_term_count",
            "rel_lhs_var_count",
            "rel_rhs_var_count"
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
        return 4