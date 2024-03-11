from abc import ABC, abstractmethod

class BaseFuncApproximator(ABC):
    @abstractmethod
    def predict_q(self, state, action):
        """
        Predict the Q-value for a given state-action pair.

        Parameters:
        - state: The current state of the environment.
        - action: The action taken in the current state.

        Returns:
        - The predicted Q-value.
        """
        pass

    @abstractmethod
    def update_q(self, state, action, target):
        """
        Update the function approximator parameters based on the target Q-value.

        Parameters:
        - state: The current state of the environment.
        - action: The action taken in the current state.
        - target: The target Q-value for the update.
        """
        pass

    @abstractmethod
    def predict_v(self, state):
        """
        Predict the V-value for a given state.

        Parameters:
        - state: The current state of the environment.

        Returns:
        - The predicted V-value.
        """
        pass

    @abstractmethod
    def update_v(self, state, target):
        """
        Update the function approximator parameters based on the target V-value.

        Parameters:
        - state: The current state of the environment.
        - target: The target V-value for the update.
        """
        pass
