import numpy as np
try:
    from .base_approx import BaseFuncApproximator, FeatureExtractor
except ImportError:
    from base_approx import BaseFuncApproximator, FeatureExtractor
import pickle
import json
import os

class LinearFuncApproximator(BaseFuncApproximator):
    def __init__(self, feature_extractor, num_features, num_actions, learning_rate=0.01, random_init=False):
        self._feature_extractor = feature_extractor  # Callback for feature extraction
        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        if random_init:
            self.q_weights = np.random.rand(num_features, num_actions)
            self.v_weights = np.random.rand(num_features)
        else:
            self.q_weights = np.zeros((num_features, num_actions))
            self.v_weights = np.zeros(num_features)
    
    @property
    def feature_extractor(self) -> FeatureExtractor:
        return self._feature_extractor

    def predict_q(self, state, action):
        features = self._feature_extractor(state)
        return np.dot(features, self.q_weights[:, action])

    def update_q(self, state, action, target):
        features = self._feature_extractor(state)
        prediction = self.predict_q(state, action)
        error = target - prediction
        self.q_weights[:, action] += self.learning_rate * error * features

    def predict_v(self, state):
        features = self._feature_extractor(state)
        return np.dot(features, self.v_weights)

    def update_v(self, state, target):
        features = self._feature_extractor(state)
        prediction = self.predict_v(state)
        error = target - prediction
        self.v_weights += self.learning_rate * error * features
    
    def pretty_print_state(self, state):
        state_approx = self.feature_extractor.pretty_print_state(state)
        return state_approx
    
    def pretty_print_action(self, action):
        action_approx = self.feature_extractor.pretty_print_action(action)
        return action_approx
    
    def pretty_print_approximator(self):
        feature_names = self.feature_extractor.pretty_print_feature_extractor()
        # Convert the concatenated array style feature names to a list [a, b, c, ...]
        feature_names = [feature_name.strip('[ ]') for feature_name in feature_names.split(",")]
        full_msgs = []
        for feature_name, wts in zip(feature_names, self.q_weights):
            indices = list(np.nonzero(wts)[0])
            if len(indices) > 0:
                q_vals_actions = [self.pretty_print_action(action) for action in indices]
            else:
                q_vals_actions = []
            wts = [wts[i] for i in indices]
            wts_prod = [f"{wts:.6f} * {action}" for wts, action in zip(wts, q_vals_actions)]
            if len(wts_prod) == 0:
                wts_prod.append("0.0")
            msg = "(" + "+".join(wts_prod) + ") * " + feature_name
            full_msgs.append(msg)
        return " + \n".join(full_msgs)
    
    def save(self, folder):
        filename = f"{folder}/LinearFuncApproximator.npz"
        feature_extractor_filename = f"{folder}/FeatureExtractor.pkl"
        with open(feature_extractor_filename, 'wb') as f:
            pickle.dump(self._feature_extractor, f)
        settings = {
            "num_features": self.num_features,
            "num_actions": self.num_actions,
            "learning_rate": self.learning_rate,
            "random_init": False,
            "type": "LinearFuncApproximator"
        }
        settings_filename = f"{folder}/Settings.json"
        with open(settings_filename, 'w') as f:
            json.dump(settings, f)
        np.savez(filename, q_weights=self.q_weights, v_weights=self.v_weights)

    def load(folder):
        filename = os.path.join(folder, "LinearFuncApproximator.npz")  #f"{folder}/LinearFuncApproximator.npz"
        settings_filename = os.path.join(folder, "Settings.json") # f"{folder}/Settings.json"
        with open(settings_filename, 'r') as f:
            settings = json.load(f)
        with open(os.path.join(folder, "FeatureExtractor.pkl"), 'rb') as f:
            feature_extractor = pickle.load(f)
        q_weights = np.load(filename)['q_weights']
        v_weights = np.load(filename)['v_weights']
        func_approximator = LinearFuncApproximator(
            feature_extractor, 
            settings["num_features"], 
            settings["num_actions"], 
            settings["learning_rate"], 
            settings["random_init"])
        func_approximator.q_weights = q_weights
        func_approximator.v_weights = v_weights
        return func_approximator
    
    def __str__(self) -> str:
        return f"LinearFuncApproximator(num_features={self.num_features}, num_actions={self.num_actions}, learning_rate={self.learning_rate})"