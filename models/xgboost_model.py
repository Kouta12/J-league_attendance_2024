import xgboost as xgb
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, params: dict):
        self.params = params
        self.model = None

    def trian(self, X, y):
        train_data = xgb.DMatrix(X, y)
        self.model = xgb.train(
            self.params,
            train_data
        )

    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X))
    
    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model = xgb.Booster()
        self.model.load_model(path)

        