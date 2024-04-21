import sys
root_dir = __file__.split('symrl')[0]
if root_dir not in sys.path:
    sys.path.append(root_dir)
from abc import ABC, abstractmethod
from symrl.func_approximator.base_approx import BaseFuncApproximator

class BaseAlgo(ABC):
    def __init__(self, func_approximator : BaseFuncApproximator):
        self.func_approximator = func_approximator

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        pass
