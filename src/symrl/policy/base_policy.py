from abc import ABC, abstractmethod

class BasePolicy(ABC):
    @abstractmethod
    def select_action(self, observation):
        pass

    @abstractmethod
    def pretty_print_state(self, state):
        pass

    @abstractmethod
    def pretty_print_action(self, action):
        pass

    @abstractmethod
    def pretty_print_policy(self):
        pass

    @abstractmethod
    def save(self, filename):
        pass