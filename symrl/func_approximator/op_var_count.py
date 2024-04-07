from symrl.environment.sympy_env import SympyEnv
from symrl.func_approximator.base_approx import FeatureExtractor
import numpy as np


class OpVarCountFeatureExtractor(FeatureExtractor):
    def __init__(self, env: SympyEnv):
        self.env = env

    def __call__(self, observation):
        lhs_term_cnt, rhs_term_cnt = SympyEnv.get_lhs_rhs_term_count(observation)
        lhs_op_cnt, rhs_op_cnt = SympyEnv.get_lhs_rhs_op_count(observation)
        all_count = lhs_term_cnt + rhs_term_cnt + lhs_op_cnt + rhs_op_cnt
        if all_count == 0:
            all_count = 1 
        all_lhs = (lhs_term_cnt + lhs_op_cnt)
        all_rhs = (rhs_term_cnt + rhs_op_cnt)
        lhs = 1 if all_lhs >= all_rhs else 0
        rhs = 1 if all_rhs >= all_lhs else 0
        lhs_1 = all_lhs
        rhs_1 = all_rhs
        all_lhs_var_count, all_rhs_var_count = SympyEnv.get_lhs_rhs_var_count(observation)
        lhs_var_count = 1 if all_lhs_var_count >= all_rhs_var_count else 0
        rhs_var_count = 1 if all_rhs_var_count >= all_lhs_var_count else 0
        lhs_2 = all_lhs_var_count
        rhs_2 = all_rhs_var_count
        return np.array([lhs, rhs, lhs_1, rhs_1, lhs_var_count, rhs_var_count, lhs_2, rhs_2])

    def pretty_print_feature_extractor(self) -> str:
        feature_names = [
            "lhs_term_count_gt_rhs_term_count", 
            "rhs_term_count_gt_lhs_term_count", 
            "lhs_term_count", 
            "rhs_term_count", 
            "lhs_var_count_gt_rhs_var_count", 
            "rhs_var_count_gt_lhs_var_count", 
            "lhs_var_count", 
            "rhs_var_count"
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
        return 8