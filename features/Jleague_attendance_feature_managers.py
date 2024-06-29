import json
import os
from typing import Any, Dict, Optional, Tuple
import csv
import time
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
    


"""
--------------------------------------------------------
▼                  特徴量の作成 ここから                   ▼
--------------------------------------------------------           
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
    
class VenueLabel(FeatureBase):
    def generate_feature(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        train_feature, test_feature = pd.DataFrame(), pd.DataFrame()

        file_path = BASE_DIR + "/function_data/sorted_venue.csv"
        sorted_venue_df = pd.read_csv(file_path)
        sorted_venue_mapping = {col: idx for idx, col in enumerate(sorted_venue_df["venue"])}

        train_feature["VenueLabel"] = train_data["venue"].map(sorted_venue_mapping)
        test_feature["VenueLabel"] = test_data["venue"].map(sorted_venue_mapping)

        create_memo(
            col_name="VenueLabel",
            description="会場を定員で昇順にマッピングする",
            data_type="int64",
            possible_values={
                'いわぎんスタジアム': 0,
                'ハワイアンズスタジアムいわき': 1,
                '愛鷹広域公園多目的競技場': 2,
                'プライフーズスタジアム': 3,
                '今治里山スタジアム': 4,
                'いちご宮崎新富サッカー場': 5,
                'ロートフィールド奈良': 6,
                'いわきグリーンフィールド': 7,
                'とうほう・みんなのスタジアム': 8,
                '相模原ギオンスタジアム': 9,
                '味の素フィールド西が丘': 10,
                '藤枝総合運動公園サッカー場': 11,
                'ケーズデンキスタジアム水戸': 12,
                'タピック県総ひやごんスタジアム': 13,
                '金沢ゴーゴーカレースタジアム': 14,
                'Axisバードスタジアム': 15,
                '鹿児島県立鴨池陸上競技場': 16,
                '白波スタジアム': 17,
                'コカ・コーラウエスト広島スタジアム': 18,
                '熊本市水前寺競技場': 19,
                '三協フロンテア柏スタジアム': 20,
                '維新みらいふスタジアム': 21,
                'ヤマハスタジアム(磐田)': 22,
                '正田醤油スタジアム群馬': 23,
                'ミクニワールドスタジアム北九州': 24,
                '町田GIONスタジアム': 25,
                '日立柏サッカー場': 26,
                'レモンガススタジアム平塚': 27,
                'Shonan BMW スタジアム平塚': 28,
                '熊谷スポーツ文化公園陸上競技場': 29,
                '横浜市三ツ沢公園球技場': 30,
                'ニッパツ三ツ沢球技場': 31,
                'シティライトスタジアム': 32,
                '町田市立陸上競技場': 33,
                'NACK5スタジアム大宮': 34,
                '長野Uスタジアム': 35,
                '大阪長居第2陸上競技場': 36,
                'ShonanBMWスタジアム平塚': 37,
                '平塚競技場': 38,
                'JIT リサイクルインク スタジアム': 39,
                '山梨県小瀬スポーツ公園陸上競技場': 40,
                '大分市営陸上競技場': 41,
                '岐阜メモリアルセンター長良川競技場': 42,
                '山梨中銀スタジアム': 43,
                '佐賀県総合運動場陸上競技場': 44,
                '駒沢オリンピック公園総合運動場陸上競技場': 45,
                '鳴門・大塚スポーツパークポカリスエットスタジアム': 46,
                '鳴門・大塚スポーツパーク ポカリスエットスタジアム': 47,
                '栃木県グリーンスタジアム': 48,
                'ソユースタジアム': 49,
                '富山県総合運動公園陸上競技場': 50,
                'フクダ電子アリーナ': 51,
                'ユアテックスタジアム仙台': 52,
                'IAIスタジアム日本平': 53,
                '柏の葉公園総合競技場': 54,
                '名古屋市瑞穂球技場': 55,
                '名古屋市瑞穂陸上競技場': 56,
                '日本平スタジアム': 57,
                'アウトソーシングスタジアム日本平': 58,
                'トランスコスモススタジアム長崎': 59,
                '石川県西部緑地公園陸上競技場': 60,
                'サンプロ アルウィン': 61,
                '松本平広域公園総合球技場': 62,
                'たけびしスタジアム京都': 63,
                '京都市西京極総合運動公園陸上競技場兼球技場': 64,
                'NDソフトスタジアム山形': 65,
                '駅前不動産スタジアム': 66,
                '札幌厚別公園競技場': 67,
                'ニンジニアスタジアム': 68,
                '万博記念競技場': 69,
                '浦和駒場スタジアム': 70,
                'さいたま市浦和駒場スタジアム': 71,
                'ベスト電器スタジアム': 72,
                'レベルファイブスタジアム': 73,
                'サンガスタジアム by KYOCERA': 74,
                '北上総合運動公園北上陸上競技場': 75,
                '東平尾公園博多の森球技場': 76,
                'Pikaraスタジアム': 77,
                'セービング陸上競技場': 78,
                '下関市営下関陸上競技場': 79,
                'ベストアメニティスタジアム': 80,
                'ヨドコウ桜スタジアム': 81,
                'キンチョウスタジアム': 82,
                'カンセキスタジアムとちぎ': 83,
                '維新百年記念公園陸上競技場': 84,
                '等々力陸上競技場': 85,
                '東大阪市花園ラグビー場': 86,
                'Uvanceとどろきスタジアム by Fujitsu': 87,
                'パロマ瑞穂スタジアム': 88,
                'エディオンピースウイング広島': 89,
                'ホームズスタジアム神戸': 90,
                'ノエビアスタジアム神戸': 91,
                '東平尾公園博多の森陸上競技場': 92,
                'えがお健康スタジアム': 93,
                '熊本県民総合運動公園陸上競技場': 94,
                '九州石油ドーム': 95,
                'レゾナックドーム大分': 96,
                '神戸総合運動公園ユニバー記念競技場': 97,
                '広島ビッグアーチ': 98,
                'エディオンスタジアム広島': 99,
                '県立カシマサッカースタジアム': 100,
                '札幌ドーム': 101,
                'パナソニックスタジアム吹田': 102,
                'パナソニック スタジアム 吹田': 103,
                '昭和電工ドーム大分': 104,
                '大分銀行ドーム': 105,
                '市立吹田サッカースタジアム': 106,
                '東北電力ビッグスワンスタジアム': 107,
                '新潟スタジアム': 108,
                'デンカビッグスワンスタジアム': 109,
                '豊田スタジアム': 110,
                '味の素スタジアム': 111,
                '大阪長居スタジアム': 112,
                '宮城スタジアム': 113,
                'ヤンマースタジアム長居': 114,
                'エコパスタジアム': 115,
                '静岡スタジアムエコパ': 116,
                '国立競技場': 117,
                '埼玉スタジアム2002': 118,
                '日産スタジアム': 119
            }
        )  
        return (train_feature, test_feature)
    
class Temperature(FeatureBase):
    def generate_feature(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        train_feature, test_feature = pd.DataFrame(), pd.DataFrame()

        train_feature["Temperature"] = train_data["temperature"]
        test_feature["Temperature"] = test_data["temperature"]

        create_memo(
            col_name="Temperature",
            description="気温（加工なし）",
            data_type="float64",
        )
        return (train_feature, test_feature)
    
class StandardizedTemperature(FeatureBase):
    def generate_feature(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        train_feature, test_feature = pd.DataFrame(), pd.DataFrame()

        sc = StandardScaler()
        X_train = np.array(train_data["temperature"]).reshape(-1, 1)
        X_test = np.array(test_data["temperature"]).reshape(-1, 1)
        sc.fit(X_train)
        train_feature["StandardizedTemperature"] = sc.transform(X_train).ravel()
        test_feature["StandardizedTemperature"] = sc.transform(X_test).ravel()

        create_memo(
            col_name="StandardizedTemperature",
            description="標準化された気温",
            data_type="float64"
        )
        return (train_feature, test_feature)
    

"""
--------------------------------------------------------
▲                  特徴量の作成 ここまで                   ▲
--------------------------------------------------------
"""



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

        """
        --------------------------------------------------------
        ▼                  特徴量の追加 ここから                   ▼
        --------------------------------------------------------
        """
        # 個々の特徴量クラスのインスタンスを作成し、特徴量を生成
        feature_classes = [
            MatchDay, 
            KickoffTime, 
            HolidayFlag,
            VenueLabel,
            Temperature,
            StandardizedTemperature,
        ]
        """
        --------------------------------------------------------
        ▲                  特徴量の追加 ここまで                   ▲
        --------------------------------------------------------
        """
        
        train_features, test_features = self.create_and_concat_features(
            train_feature,
            test_feature,
            feature_classes,
            train_data,
            test_data,
        )

        return (train_features, test_features)

    
if __name__=="__main__":
    data_dir = BASE_DIR + "/features/feature_data"
    train_path = BASE_DIR + "/function_data/train.csv"
    test_path = BASE_DIR + "/function_data/test.csv"

    j_league_attendance = JLeagueAttendance(data_dir)
    train_feature, test_feature = j_league_attendance.create_feature(
        train_path=train_path,
        test_path=test_path
    )