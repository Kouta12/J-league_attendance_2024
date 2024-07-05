import lightgbm as lgb
from .base_model import BaseModel
from sklearn.metrics import mean_squared_error
import numpy as np

class LightGBMModel(BaseModel):
    def __init__(self, params: dict):
        self.params = params
        self.model = None
        
        self.evaluation_results = None

    def train(self, X_tr, y_tr, X_va, y_va):
        train_dataset = lgb.Dataset(X_tr, y_tr)
        eval_dataset = lgb.Dataset(X_va, y_va, reference=train_dataset)
        evaluation_results = {}
        self.model = lgb.train(
            self.params, 
            train_set=train_dataset,
            valid_sets=[train_dataset, eval_dataset],
            valid_names=['train', 'valid'],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(100, verbose=True),
                lgb.log_evaluation(100)
            ],
            evals_result=evaluation_results  # 評価結果を格納
        )
        
        # 評価結果を self.evaluation_results に格納
        self.evaluation_results = evaluation_results


    def train_with_callback(self, X_train, y_train, X_val, y_val, best_params):
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        iterations = []
        train_scores = []
        val_scores = []

        def callback(env):
            iteration = env.iteration
            if iteration % 10 == 0 or iteration == env.end_iteration - 1:  # 10イテレーションごとまたは最後のイテレーションで記録
                iterations.append(iteration)
                y_train_pred = env.model.predict(X_train, num_iteration=iteration)
                y_val_pred = env.model.predict(X_val, num_iteration=iteration)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                train_scores.append(train_rmse)
                val_scores.append(val_rmse)

        self.model = lgb.train(
            params=best_params,
            train_set=train_data,
            num_boost_round=10000,
            valid_sets=[train_data, val_data],
            callbacks=[callback]
        )

        return iterations, train_scores, val_scores
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model = lgb.Booster(model_file=path)

        