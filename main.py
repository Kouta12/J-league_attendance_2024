import os
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt


from typing import Tuple, Any
from models.base_model import BaseModel

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
        params: dict[str, Any]
):
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

    info_data[f"round_{n_len}"] = run_info
    print(f"'{run_name}' - round_{n_len} の情報を追加しました。")
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(info_data, f, indent=4, ensure_ascii=False)


def shap_summary_plot(run_name: str, models: list[BaseModel], X: pd.DataFrame):
    # すべてのフォールドのモデルを使ってSHAP値を計算
    shap_values = (shap.Explainer(model)(X) for model in models)

    # SHAP値を平均化
    shap_values_avg = sum(shap_values) / len(shap_values)


    # 特徴量の重要度を計算
    importance_df = pd.DataFrame(list(zip(X.columns, np.abs(shap_values_avg.values).mean(0))), 
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

