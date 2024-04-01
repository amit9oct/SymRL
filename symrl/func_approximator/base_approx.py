from abc import ABC, abstractmethod, abstractproperty
import json

class FeatureExtractor(ABC):
    @abstractmethod
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)
    
    @abstractmethod
    def pretty_print_state(self, state) -> str:
        pass

    @abstractmethod
    def pretty_print_action(self, action) -> str:
        pass

    @abstractmethod
    def pretty_print_feature_extractor(self) -> str:
        pass

class BaseFuncApproximator(ABC):

    @abstractproperty
    def feature_extractor(self) -> FeatureExtractor:
        pass

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

    @abstractmethod
    def save(self, folder):
        """
        Save the function approximator to a file.

        Parameters:
        - folder: The folder where the function approximator should be saved.
        """
        pass
    
    @abstractmethod
    def pretty_print_state(self, state) -> str:
        """
        Pretty print the state.

        Parameters:
        - state: The current state of the environment.

        Returns:
        - The pretty printed state.
        """
        pass

    @abstractmethod
    def pretty_print_action(self, action) -> str:
        """
        Pretty print the action.

        Parameters:
        - action: The action taken in the current state.

        Returns:
        - The pretty printed action.
        """
        pass

    @abstractmethod
    def pretty_print_approximator(self) -> str:
        """
        Pretty print the function approximator.

        Returns:
        - The pretty printed function approximator.
        """
        pass

    def load(self, folder):
        try:
            from .linear_fun_approx import LinearFuncApproximator
            from .nn_fun_approx import NeuralFuncApproximator
        except ImportError:
            from linear_fun_approx import LinearFuncApproximator
            from nn_fun_approx import NeuralFuncApproximator
        with open(f"{folder}/Settings.json", 'r') as f:
            settings = json.load(f)
        if settings["type"] == "LinearFuncApproximator":
            func_approximator = LinearFuncApproximator.load(folder)
        elif settings["type"] == "NeuralFuncApproximator":
            func_approximator = NeuralFuncApproximator.load(folder)
        else:
            raise ValueError(f"Unknown function approximator type: {settings['type']}")
        return func_approximator
        
