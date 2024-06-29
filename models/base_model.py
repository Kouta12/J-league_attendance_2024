from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
import numpy as np
from sklearn.metrics import mean_squared_error

class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y):
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

    def create_callback(
            self,
            X_train,
            y_train,
            X_val,
            y_val, 
    ) -> Tuple[Callable, List[int], List[float], List[float]]:
        iterations = []
        train_scores = []
        val_scores = []

        def callback(env):
            iteration = env.iteration if hasattr(env, "iteration") else len(iteration)

            # トレーニングデータでの評価
            y_train_pred = self.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            # バリデーションデータでの評価
            y_val_pred = self.predict(X_val)
            val_rsme = np.sqrt(mean_squared_error(y_val, y_val_pred))

            iterations.append(iteration)
            train_scores.append(train_rmse)
            val_scores.append(val_rsme)

        return callback, iterations, train_scores, val_scores