import json
import os
from typing import Any, Dict, Optional, Tuple
import csv
import time
from pathlib import Path
from contextlib import contextmanager
import pandas as pd

# ▼ペアレントディレクトリの定義
BASE_DIR = str(Path(os.path.abspath('')))

def create_memo(
    col_name: str, 
    description: str, 
    data_type: str, 
    possible_values: Optional[Any] = None
):
    """
    特徴量に関する詳細な情報をJSONファイルに書き込む

    :param col_name: 特徴量の名前
    :param description: 特徴量の説明
    :param data_type: 特徴量のデータ型
    :param possible_values: 取りうる値の説明（オプション）
    """
    file_path = os.path.join(BASE_DIR, "features", "_features_memo.json")
    
    # ファイルが存在しない場合、空の辞書で初期化
    if not os.path.isfile(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
    
    # 既存のデータを読み込む
    with open(file_path, "r", encoding="utf-8") as f:
        memo_data = json.load(f)
    
    # 新しい特徴量情報を作成
    feature_info = {
        "description": description,
        "data_type": data_type,
        "possible_values": possible_values
    }
    
    # 新しい特徴量が既に存在しない場合のみ追加、存在する場合は更新
    if col_name not in memo_data:
        memo_data[col_name] = feature_info
        print(f"特徴量 '{col_name}' の情報を追加しました。")
    else:
        memo_data[col_name].update(feature_info)
        print(f"特徴量 '{col_name}' の情報を更新しました。")
    
    # 更新されたデータを書き込む
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(memo_data, f, indent=4, ensure_ascii=False)

# ▼feather形式のファイルを読み込む
def load_feather(feats: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dfs = [pd.read_feather(f'{BASE_DIR}/features/feature_data/{f}_train.feather') for f in feats]
    X_train = pd.concat(dfs, axis=1)
    dfs = [pd.read_feather(f'{BASE_DIR}/features/feature_data/{f}_test.feather') for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return (X_train, X_test)

# タイマー
@contextmanager
def timer(name: str):
    t0 = time.time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")



class FeatureBase:
    """
    Jリーグの観客数を予測するコンペの特徴量を管理するクラス
    """

    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

    def create_feature(
            self, 
            train_data: pd.DataFrame, 
            test_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        with timer(name=self.__class__.__name__):
            train_feature, test_feature = self.generate_feature(train_data, test_data)
            self.save_feature(train_feature, test_feature)
        return (train_data, test_data)

    def generate_feature(
            self, 
            train_data: pd.DataFrame, 
            test_data: pd.DataFrame
    ):
        return NotImplementedError
    
    def save_feature(
            self,
            train_feature: pd.DataFrame,
            test_feature: pd.DataFrame,
    ):
        train_feature.to_feather(os.path.join(self.data_dir, f"{self.__class__.__name__}_train.feather"))
        test_feature.to_feather(os.path.join(self.data_dir, f"{self.__class__.__name__}_test.feather"))



class JLeagueAttendance:
    """
    コンペの特徴量を管理するクラス
    """
    def __init__(
            self,
            data_dir,
    ):
        self.data_dir = data_dir

    def create_and_concat_features(
            self,
            train_features: pd.DataFrame,
            test_features: pd.DataFrame,
            feature_classes: list[FeatureBase],
            train_data: pd.DataFrame,
            test_data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for feature_class in feature_classes:
            feature_instance = feature_class(self.data_dir)
            train_feature, test_feature = feature_instance.create_feature(
                train_data, 
                test_data
            )
            train_features = pd.concat(
                [train_features, train_feature],
                axis=1
            )
            test_features = pd.concat(
                [test_features, test_feature],
                axis=1
            )
        return (train_features, test_features)
    
    def create_feature(
            self,
            train_path: str,
            test_path: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        train_feature = pd.DataFrame()
        test_feature = pd.DataFrame()

        # 個々の特徴量クラスのインスタンスを作成し、特徴量を生成
        feature_classes = [
            MatchDay, KickoffTime, HolidayFlag
        ]
        
        train_features, test_features = self.create_and_concat_features(
            train_feature,
            test_feature,
            feature_classes,
            train_data,
            test_data
        )

        return (train_features, test_features)
    

"""
▼ 特徴量の作成 ▼             
"""

class MatchDay(FeatureBase):
    def generate_feature(
            self, 
            train_data: pd.DataFrame, 
            test_data: pd.DataFrame
    ):
        train_feature, test_feature = pd.DataFrame(), pd.DataFrame()

        # 型を変えるカラムを選択
        train_feature["MatchDay"] = pd.to_datetime(train_data["match_date"])
        test_feature["MatchDay"] = pd.to_datetime(test_data["match_date"])

        create_memo(
            col_name="MatchDay",
            description="試合日",
            data_type="datetime64[ns]"
        )
        return train_feature, test_feature
    
class KickoffTime(FeatureBase):
    def generate_feature(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        train_feature, test_feature = pd.DataFrame(), pd.DataFrame()

        # キックオフの時刻を整数にする
        train_feature["KickoffTime"] = pd.to_datetime(train_data["kick_off_time"]).dt.hour
        test_feature["KickoffTime"] = pd.to_datetime(test_data["kick_off_time"]).dt.hour

        create_memo(
            col_name="KickoffTime",
            description="キックオフの時間を整数にした",
            data_type="int64"
        )
        return train_feature, test_feature
    
class HolidayFlag(FeatureBase):
    def create_holiday_flag(self, x: pd.DataFrame):
        if (x["match_weekday"] in [5, 6]) | (pd.isna(x["description"]) == False):
            return 1
        else:
            return 0

    def generate_feature(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        train_feature, test_feature = pd.DataFrame(), pd.DataFrame()

        # 日付から曜日を求める
        train_data["match_weekday"] = pd.to_datetime(train_data["match_date"]).dt.weekday
        test_data["match_weekday"] = pd.to_datetime(train_data["match_date"]).dt.weekday

        train_feature["holiday_flag"] = train_data.apply(
            self.create_holiday_flag
            ,axis=1
        )
        test_feature["holiday_flag"] = test_data.apply(
            self.create_holiday_flag,
            axis=1
        )

        create_memo(
            col_name="HolidayFlag",
            description="休日のフラグ",
            data_type="int64",
            possible_values={
                0: "平日",
                1: "休日",
            }
        )
        return train_feature, test_feature
    


    
if __name__=="__main__":
    data_dir = BASE_DIR + "/features/feature_data"
    train_path = BASE_DIR + "/function_data/train.csv"
    test_path = BASE_DIR + "/function_data/test.csv"

    j_league_attendance = JLeagueAttendance(data_dir)
    train_feature, test_feature = j_league_attendance.create_feature(
        train_path=train_path,
        test_path=test_path
    )