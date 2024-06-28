from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel

def create_model(model_name: str, params: dict):
    if model_name == "lightgbm":
        return LightGBMModel(params)
    elif model_name == "xgboost":
        return XGBoostModel(params)
    else:
        return ValueError(f"未設定のモデル {model_name}")