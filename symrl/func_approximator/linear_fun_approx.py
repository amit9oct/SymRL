import numpy as np
try:
    from .base_approx import BaseFuncApproximator
except ImportError:
    from base_approx import BaseFuncApproximator

class LinearFuncApproximator(BaseFuncApproximator):
    def __init__(self, feature_extractor, num_features, num_actions, learning_rate=0.01, random_init=False):
        self.feature_extractor = feature_extractor  # Callback for feature extraction
        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        if random_init:
            self.q_weights = np.random.rand(num_features, num_actions)
            self.v_weights = np.random.rand(num_features)
        else:
            self.q_weights = np.zeros((num_features, num_actions))
            self.v_weights = np.zeros(num_features)

    def predict_q(self, state, action):
        features = self.feature_extractor(state)
        return np.dot(features, self.q_weights[:, action])

    def update_q(self, state, action, target):
        features = self.feature_extractor(state)
        prediction = self.predict_q(state, action)
        error = target - prediction
        self.q_weights[:, action] += self.learning_rate * error * features

    def predict_v(self, state):
        features = self.feature_extractor(state)
        return np.dot(features, self.v_weights)

    def update_v(self, state, target):
        features = self.feature_extractor(state)
        prediction = self.predict_v(state)
        error = target - prediction
        self.v_weights += self.learning_rate * error * features