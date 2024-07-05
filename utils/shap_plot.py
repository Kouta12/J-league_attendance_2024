import shap
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from models.base_model import BaseModel
from typing import List


def shap_summary_plot(run_name: str, BASE_DIR: str, model: BaseModel, X: pd.DataFrame) -> None:
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

def shap_summary_plot_ensemble(run_name: str, BASE_DIR: str, models: List[BaseModel], X: pd.DataFrame) -> None:
    # 全てのフォールドのモデルを使ってSHAP値を計算
    shap_values = [shap.Explainer(model)(X) for model in models]

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
    file_path = os.path.join(BASE_DIR, f"logs/{run_name}/{run_name}_shap_importance_ensemble.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()