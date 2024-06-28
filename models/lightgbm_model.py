import lightgbm as lgb
from .base_model import BaseModel

class LightGBMModel(BaseModel):
    def __init__(self, params: dict):
        self.params = params
        self.model = None

    def train(self, X, y):
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            self.params, 
            train_set=train_data
        )
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model = lgb.Booster(model_file=path)

        