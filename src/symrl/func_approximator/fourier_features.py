from symrl.environment.sympy_env import SympyEnv
import numpy as np


class FourierFeatureExtractor:
    def __init__(self, env: SympyEnv, n_features=10, sigma=1.0):
        self.n_features = n_features
        self.sigma = sigma # Scale of the random transformation
        # Assuming 4 base features: ["lhs_term_count", "rhs_term_count", "lhs_var_count", "rhs_var_count"]
        # Initialize random weights for each base feature for the transformation
        self.weights = np.random.randn(4, n_features) * sigma
        self.original_feature_names = ["lhs_term_count", "rhs_term_count", "lhs_var_count", "rhs_var_count"]
        self.env = env

    def __call__(self, observation):
        lhs_term_cnt, rhs_term_cnt = SympyEnv.get_lhs_rhs_term_count(observation)
        lhs_op_cnt, rhs_op_cnt = SympyEnv.get_lhs_rhs_op_count(observation)
        lhs, rhs = (lhs_term_cnt + lhs_op_cnt), (rhs_term_cnt + rhs_op_cnt)
        lhs_var_count, rhs_var_count = SympyEnv.get_lhs_rhs_var_count(observation)
        # Ensure observation is a numpy array
        observation = np.array([lhs, rhs, lhs_var_count, rhs_var_count])
        # Normalize the observation to ensure the Fourier features are bounded
        norm_obs = (observation - np.mean(observation)) / (np.std(observation) + 1e-7)
        # Compute the transformation
        transformed_obs = np.dot(norm_obs, self.weights)
        # Apply the sinusoidal function to get the Fourier features
        fourier_features = np.cos(transformed_obs)
        return fourier_features.flatten()

    def pretty_print_feature_extractor(self):
        feature_descriptions = []
        for original_feature in self.original_feature_names:
            for i in range(self.n_features):
                feature_descriptions.append(f"{original_feature}_fourier_feature_{i}")
        return ", ".join(feature_descriptions)

    def pretty_print_state(self, state) -> str:
        state_str = str(self(state))
        return state_str

    def pretty_print_action(self, action) -> str:
        action_str = self.env.action_space.actions[action]
        return action_str
    
    @property
    def num_features(self):
        return self.n_features * 4