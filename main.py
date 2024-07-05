import os
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt
import optuna
from optuna.trial import Trial

from typing import Tuple, Any, Literal, Dict, List
from numpy import ndarray
from models.base_model import BaseModel

from models.model_factory import create_model
from utils.plotting import plot_learning_curve, plot_learning_curve_ensemble
from utils.shap_plot import shap_summary_plot, shap_summary_plot_ensemble


# ▼ペアレントディレクトリの定義
BASE_DIR = str(Path(os.path.abspath('')))

# ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# feather形式のファイルを読み込む
def load_feather(feats: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dfs = [pd.read_feather(f'{BASE_DIR}/features/feature_data/{f}_train.feather') for f in feats]
    X_train = pd.concat(dfs, axis=1)
    dfs = [pd.read_feather(f'{BASE_DIR}/features/feature_data/{f}_test.feather') for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return (X_train, X_test)

# 使用したモデル名・特徴量・ハイパーパラメータを保存する　
def save_run_info(
        run_name: str, 
        model_name: str,
        features: list[str], 
        params: dict[str, Any],
        final: bool = False,
) -> None:
    file_path = os.path.join(BASE_DIR, f"logs/{run_name}/{run_name}_info.json")
    
    # ファイルが存在しない場合、空の辞書で初期化
    if not os.path.isfile(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

    # 既存のデータを読み込む
    with open(file_path, "r", encoding="utf-8") as f:
        info_data = json.load(f)

    # 既存のデータから回数を取得
    n_len = len(info_data) + 1
    # 情報を書き込む
    run_info = {
        "model_name": model_name,
        "features": features,
        "params": params
    }

    if final:
        info_data["final"] = run_info
        print(f"'{run_name}' - round_{n_len} の情報を追加しました。")


    else:
        info_data[f"round_{n_len}"] = run_info
        print(f"'{run_name}' - round_{n_len} の情報を追加しました。")
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(info_data, f, indent=4, ensure_ascii=False)




class JLeagueAttendanceModel:
    def __init__(
            self,
            run_name: str,
            model_naeme : Literal["lightgbm", "xgboost"],
            n_splits : int = 5,
    ):
        self.run_name = run_name
        self.model_name = model_naeme
        self.n_splits = n_splits
        self.optuna_params = None

        self.model = None
        self.models: List[BaseModel] = []
        self.y_test_pred = None
        self.log_dir = f"{BASE_DIR}/logs/{run_name}"

        # ログファイル自動生成
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        # ファイルハンドラの設定
        file_handler = logging.FileHandler(self.log_dir + "/general.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def train(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            params: Dict[str, Any],
    ) -> None:

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=777)

        evaluation_results_list = []
        valid_scores = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            model = create_model(
                model_name=self.model_name,
                params=params
            )
            model.train(X_tr, y_tr, X_va=X_va, y_va=y_va)

            y_va_pred = model.predict(X_va)
            valid_score = np.sqrt(mean_squared_error(y_va, y_va_pred))

            self.models.append(model)
            valid_scores.append(valid_score)
            evaluation_results_list.append(model.evaluation_results)


            logger.info(f"{self.run_name} - Fold {fold+1}/{self.n_splits} - score {valid_score:.2f}")


        logger.info(f"{self.run_name} - end training cv - score {np.mean(valid_scores):.2f}")

        save_run_info(
            run_name=self.run_name,
            model_name=self.model_name,
            features=X.columns.to_list(),
            # ベストパラメータにしたい
            params=params
        )

        # shap_summary_plot_ensemble(
        #     run_name=run_name,
        #     BASE_DIR=BASE_DIR,
        #     models=self.models,
        #     X=X
        # )

        plot_learning_curve_ensemble(
            run_name=run_name,
            BASE_DIR=BASE_DIR,
            eval_results_list=evaluation_results_list,
        )




    def cross_validate(self, X: pd.DataFrame, y: pd.DataFrame, params: Dict[str, Any]) -> float:
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=777)

        valid_scores = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            model = create_model(
                model_name=self.model_name,
                params=params
            )
            model.train(X_tr, y_tr, X_va, y_va)

            y_va_pred = model.predict(X_va)
            valid_score = np.sqrt(mean_squared_error(y_va, y_va_pred))

            valid_scores.append(valid_score)

            logger.info(f"{self.run_name} - Fold {fold+1}/{self.n_splits} - score {valid_score:.2f}")

        mean_score = np.mean(valid_scores)
        logger.info(f"{self.run_name} - end training cv - score {mean_score:.2f}")

        return mean_score
    
    def objective(
            self,
            trial: Trial,
            X: pd.DataFrame,
            y: pd.DataFrame,
            params: Dict[str, Any],
    ) -> List[BaseModel]:
        current_params = params.copy()
        for param_name, param_range in self.optuna_params.items():
            if param_name in ["num_leaves", "n_estimators"]:
                current_params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
            elif param_name == "learning_rate":
                current_params[param_name] = trial.suggest_loguniform(param_name, param_range[0], param_range[1])
            else:
                current_params[param_name] = trial.suggest_uniform(param_name, param_range[0], param_range[1])

        mean_score = self.cross_validate(X, y, current_params)

        save_run_info(
            run_name=self.run_name,
            model_name=self.model_name,
            features=X.columns.tolist(),
            params=current_params
        )

        return mean_score
    
    def optimize_hyperparameters(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            params: Dict[str, Any],
            optuna_params: Dict[str, Tuple[float, float]],
            n_trials: int = 100,
    ) -> Dict[str, Any]:
        self.optuna_params = optuna_params
        
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial, X, y, params), n_trials=n_trials)
        
        best_params = {**params, **study.best_params}
        return best_params
        

    def train_final_model(
            self, 
            X: pd.DataFrame,
            y: pd.DataFrame,
            best_params: Dict[str, Any]
    ) -> None:
        model = create_model(
            model_name=self.model_name,
            params=best_params
        )

        # トレーニングデータとバリデーションデータに分割
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=777)

        iterations, train_scores, val_scores = model.train_with_callback(X_train, y_train, X_val, y_val, best_params)

        plot_path = plot_learning_curve(
            iterations=iterations,
            train_scores=train_scores,
            val_scores=val_scores,
            log_dir=f"{BASE_DIR}/logs",
            run_name=self.run_name
        )

        logger.info(f"学習曲線を保存しました - {plot_path}")
        # 最終的な性能を記録
        final_train_rmse = train_scores[-1]
        final_val_rmse = val_scores[-1]
        logger.info(f"Final Training RMSE: {final_train_rmse:.4f}")
        logger.info(f"Final Validation RMSE: {final_val_rmse:.4f}")

        self.model = model

        save_run_info(
            run_name=self.run_name,
            model_name=self.model_name,
            features=X.columns,
            params=best_params,
            final=True
        )

        shap_summary_plot(
            run_name=self.run_name,
            models=self.models,
            X=X
        )

    def predict(self, X_test: pd.DataFrame) -> ndarray:
        
        y_test_pred = self.model.predict(X_test)

        return y_test_pred
    
    def predict_ensemble(self, X_test: pd.DataFrame) -> np.ndarray:
        """全モデルの予測の平均を取る"""
        predictions = [model.predict(X_test) for model in self.models]
        return np.mean(predictions, axis=0)
        
    def create_submission(self, y_test_pred: ndarray):
        sample_submission_df = pd.read_csv(f"{BASE_DIR}/data/sample_submit.csv", header=None)
        sample_submission_df[1] = y_test_pred
        file_path = os.path.join(BASE_DIR, "submit_data", f"{self.run_name}_submit.csv")
        sample_submission_df.to_csv(file_path, index=False)

if __name__=="__main__":
    data_dir = os.path.join(BASE_DIR, "features/feature_data")
    train_path = os.path.join(BASE_DIR, "function_data/train.csv")
    test_path = os.path.join(BASE_DIR, "function_data/test.csv")



    run_name = "lgb_2024_06_29"

    features = [
        # "MatchDay", 
        "KickoffTime", 
        "HolidayFlag",
        "VenueLabel",
        "Temperature",
        "StandardizedTemperature",
    ]

    # ベースとなるパラメータを設定
    base_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        "num_iterations": 10000,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.01,  # 0.1から0.01に減少
        "lambda_l2": 0.1,
        "min_child_samples": 20,  # この行を追加
        "max_depth": -1,  # この行を追加。-1は無制限を意味します
    }

    # Optunaで最適化するパラメータの範囲を設定
    # optuna_params = {
    #     "num_leaves": (10, 1000),
    #     "learning_rate": (1e-4, 1e-1),  # 下限を1e-3から1e-4に変更
    #     "n_estimators": (100, 1000),
    # }

    optuna_params = {}

    train_feature, test_feature = load_feather(
        feats=features
    )

    y_train = pd.read_csv("data/train.csv")["attendance"]

    model = JLeagueAttendanceModel(
        run_name=run_name,
        model_naeme="lightgbm",
        n_splits=5
    )

    # best_params = model.optimize_hyperparameters(
    #     X=train_feature,
    #     y=y_train,
    #     params=base_params,
    #     optuna_params=optuna_params,
    #     n_trials=5
    # )

    # # Train final model with best parameters
    # model.train_final_model(train_feature, y_train, best_params)

    

    # シンプルなクロスバリデーション
    model.train(
        X=train_feature,
        y=y_train,
        params=base_params
    )

    y_test_pred = model.predict_ensemble(
        X_test=test_feature
    )

    # Create submission file
    model.create_submission(y_test_pred)

    logger.info("Model training and prediction completed successfully.")