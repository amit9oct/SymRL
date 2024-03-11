from abc import ABC, abstractmethod

class BasePolicy(ABC):
    @abstractmethod
    def select_action(self, observation):
        pass