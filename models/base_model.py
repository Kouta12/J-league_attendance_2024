from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
import numpy as np
from sklearn.metrics import mean_squared_error

class BaseModel(ABC):
    @abstractmethod
    def train(self, X_tr, y_tr, X_va, y_va):
        pass

    @abstractmethod
    def predict(self, X): 
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
