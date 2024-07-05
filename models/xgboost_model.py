import xgboost as xgb
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, params: dict):
        self.params = params
        self.model = None

    def train(self, X, y):
        train_data = xgb.DMatrix(X, y)
        self.model = xgb.train(
            self.params,
            train_data
        )

    def train_with_callback(self, X_train, y_train, X_val, y_val):
        train_data = xgb.DMatrix(X_train, y_train)
        val_data = xgb.DMatrix(X_val, y_val)
        callback, iterations, train_scores, val_scores = self.create_callback(X_train, y_train, X_val, y_val)
        
        self.model = xgb.train(
            self.params,
            train_data,
            evals=[(train_data, 'train'), (val_data, 'val')],
            callbacks=[callback]
        )
        
        return iterations, train_scores, val_scores

    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X))
    
    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model = xgb.Booster()
        self.model.load_model(path)

        