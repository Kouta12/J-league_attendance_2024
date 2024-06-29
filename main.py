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

from typing import Tuple, Any, Literal, Dict
from numpy import ndarray
from models.base_model import BaseModel

from models.model_factory import create_model
from utils.plotting import plot_learning_curve
from features.Jleague_attendance_feature_managers import JLeagueAttendance

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


def shap_summary_plot(run_name: str, model: BaseModel, X: pd.DataFrame) -> None:
    # 単一のモデルを使ってSHAP値を計算
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # 特徴量の重要度を計算
    importance_df = pd.DataFrame(list(zip(X.columns, np.abs(shap_values.values).mean(0))), 
                                 columns=['feature', 'importance'])
    importance_df = importance_df.sort_values('importance', ascending=False)

    # shap.Explanationオブジェクトを作成
    shap_exp = shap.Explanation(values=importance_df['importance'].values, 
                                base_values=None, 
                                data=importance_df['feature'].values, 
                                feature_names=importance_df['feature'].values)

    # 特徴量の重要度のバーグラフを作成
    fig, ax = plt.subplots(figsize=(10, len(importance_df) * 0.4))
    shap.plots.bar(shap_exp, show=False)

    # プロットを保存
    file_path = os.path.join(BASE_DIR, f"logs/{run_name}/{run_name}_shap_importance.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()


class JLeagueAttendanceModel:
    def __init__(
            self,
            run_name: str,
            model_naeme : Literal["lightgbm", "xgboost"],
            n_splits : int = 5,
    ):
        self.run_name = run_name
        self.model_name = model_naeme
        self.n_plits = n_splits

        self.model = None
        self.models = []
        self.y_test_pred = None
        self.log_dir = f"{BASE_DIR}/{run_name}"

        # ログファイル自動生成
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        # ファイルハンドラの設定
        file_handler = logging.FileHandler(self.log_dir + "/general.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


    def objective(
            self,
            trial: Trial,
            X: pd.DataFrame,
            y: pd.DataFrame,
            params: Dict[str, Any],
    ) -> float:

        kf = KFold(n_splits=self.n_plits, shuffle=True, random_state=777)

        valid_scores = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            model = create_model(
                model_name=self.model_name,
                params=self.params
            )
            model.train(X_tr, y_tr)

            y_va_pred = model.predict(X_va)
            valid_score = np.sqrt(mean_squared_error(y_va, y_va_pred))

            self.models.append(model)
            valid_scores.append(valid_score)

            logger.info(f"{self.run_name} - Fold {fold+1}/{self.n_splits} - score {valid_score:.2f}")


        logger.info(f"{self.run_name} - end training cv - score {np.mean(valid_scores):.2f}")

        save_run_info(
            run_name=self.run_name,
            model_name=self.model_name,
            features=X.columns,
            # ベストパラメータにしたい
            params=params
        )

        return np.mean(valid_score)

    def optimize_hyperparameters(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            params: Dict[str, Any],
            n_trials : int = 100, 
    ) -> Dict[str, Any]:
        
        study = optuna.create_study(direction="minimize") 
        study.optimize(
            lambda trial: self.objective(trial, X, y, params=params), 
            n_trials=n_trials
        )
        return study.best_params
    

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

        iterations, train_scores, val_scores = model.train_with_callback(X_train, y_train, X_val, y_val)

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
        
    def create_submission(self, y_test_pred: ndarray):
        sample_submission_df = pd.read_csv(f"{BASE_DIR}/data/sample_submission.csv", header=None)
        sample_submission_df[1] = y_test_pred
        file_path = os.path.join(BASE_DIR, "submit_data", f"{self.run_name}_submission.csv")
        sample_submission_df.to_csv(file_path, index=False)

if __name__=="__main__":
    data_dir = os.path.join(BASE_DIR, "features/feature_data")
    train_path = os.path.join(BASE_DIR, "function_data/train.csv")
    test_path = os.path.join(BASE_DIR, "function_data/test.csv")

    j_league_attendance = JLeagueAttendance(data_dir)
    train_feature, test_feature = j_league_attendance.create_feature(
        train_path=train_path,
        test_path=test_path
    )

    run_name = "lgb_2024_06_29"

    features = [
        "MatchDay", 
        "KickoffTime", 
        "HolidayFlag",
        "VenueLabel",
        "Temperature",
        "StandardizedTemperature",
    ]

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        "num_iterations": 10000,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        'feature_fraction': Trial.suggest_uniform('feature_fraction', 0.1, 1.0),
        "num_leaves": optuna.distributions.IntUniformDistribution(10, 1000),
        "learning_rate": optuna.distributions.LogUniformDistribution(1e-3, 0.1),
        "n_estimators": optuna.distributions.IntUniformDistribution(100, 1000),

    }


    X_train = train_feature[features]
    X_test = test_feature[features]
    y_train = pd.read_csv("data/train.csv")["attendance"]

    model = JLeagueAttendanceModel(
        run_name=run_name,
        model_naeme="lightgbm",
        n_splits=5
    )

    best_params = model.optimize_hyperparameters(
        X=X_train,
        y=y,
        params=params
    )

    # Train final model with best parameters
    model.train_final_model(X_train, y_train, best_params)

    # Make predictions on test set
    y_test_pred = model.predict(X_test)

    # Create submission file
    model.create_submission(y_test_pred)

    logger.info("Model training and prediction completed successfully.")