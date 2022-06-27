import random
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
import pickle
import os


def data_split(df, args):
    """
    :param df: 전처리된 데이터프레임을 입력받습니다.
    :param args: args.py의 하이퍼파라미터
    :return: 8:2로 split된 train, test 데이터셋을 반환합니다. Row 개수로 나눈 것이 아니라, 히트 개수를 기준으로 8:2로 나누었습니다.
    """
    test_size = int(df["히트NO."].nunique() * args["TEST_SIZE"])
    random.seed(args["RANDOM_SEED"])
    train_heats = random.sample(list(df["히트NO."].unique()), test_size)
    
    train_df = df[~df["히트NO."].isin(train_heats)].reset_index(drop=True)
    test_df = df[df["히트NO."].isin(train_heats)].reset_index(drop=True)
    
    return train_df, test_df


def get_train_test_data(train_df, test_df, args):
    """
    :param train_df: Split된 Train 데이터셋
    :param test_df: Split된 Test 데이터셋
    :param args: args.py의 하이퍼파라미터
    :return: 모델 학습 및 검증이 가능한 형태의 train, test 데이터셋을 반환합니다(x_train, y_train, x_test, y_test).
    이때, Train데이터셋 각 칼럼별 평균 및 표준편차를 계산하여 scaler.pkl에 저장하고, 이후 Test셋 스케일링에 사용합니다.
    """
    train_df2 = train_df.drop(["히트NO.", "총사용량"], axis=1)
    test_df2 = test_df.drop(["히트NO.", "총사용량"], axis=1)

    x_train = train_df2.drop(["EBT_OPEN"], axis=1)
    y_train = train_df2["EBT_OPEN"]   
    
    x_test = test_df2.drop(["EBT_OPEN"], axis=1)
    y_test = test_df2["EBT_OPEN"]   
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    if not os.path.isdir(args["SCALER_PATH"]):
        os.mkdir(args["SCALER_PATH"])    
    
    pickle.dump(scaler, open(args["SCALER_PATH"]+"/scaler.pkl", 'wb'))
    print("Scaler saved ...")
    return x_train, y_train, x_test, y_test


def get_train_data(train_df, args):
    """
    :param train_df: 전처리된 Train 데이터셋
    :param args: args.py의 하이퍼파라미터
    :return: 모델 학습이 가능한 형태의 train 데이터셋을 반환합니다(x_train, y_train). Train데이터셋 각 칼럼별 평균 및 표준편차를
     계산하여 scaler.pkl에 저장합니다.
    """
    train_df2 = train_df.drop(["히트NO.", "총사용량"], axis=1)
    x_train = train_df2.drop(["EBT_OPEN"], axis=1)
    y_train = train_df2["EBT_OPEN"]  
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    
    if not os.path.isdir(args["SCALER_PATH"]):
        os.mkdir(args["SCALER_PATH"])    
        
    pickle.dump(scaler, open(args["SCALER_PATH"]+"/scaler.pkl", 'wb'))
    print("Scaler saved ...")
    
    return x_train, y_train


def get_test_data(test_df, args):
    """
    :param test_df: 전처리된 Test 데이터셋
    :param args: args.py의 하이퍼파라미터
    :return: 검증 가능한 형태의 Test 데이터셋을 반환합니다(x_test, y_test). Train셋 학습에 사용된 Scaler를 불러와 Test셋 스케일링을 합니다.
    """
    test_df2 = test_df.drop(["히트NO.", "총사용량"], axis=1)
    x_test = test_df2.drop(["EBT_OPEN"], axis=1)
    y_test = test_df2["EBT_OPEN"]   
    scaler = pickle.load(open(args["SCALER_PATH"]+"/scaler.pkl", 'rb'))
    print("Scaler loaded ...")
    x_test = scaler.transform(x_test)
    
    return x_test, y_test


def get_models(args):
    """
    :param args: args.py의 하이퍼파라미터
    :return: 하이퍼파라미터가 설정된 모델들을 리스트에 저장하여 반환합니다.
    """
    ran = RandomForestClassifier(random_state=args["RANDOM_SEED"],
                                 max_depth=None,
                                 max_features=0.5,
                                 min_samples_leaf=6,
                                 min_samples_split=2,
                                 n_estimators=200)
    ext = ExtraTreesClassifier(random_state=args["RANDOM_SEED"],
                               max_depth=None,
                               max_features=0.5,
                               min_samples_leaf=2,
                               min_samples_split=10,
                               n_estimators=75)
    lgbm = LGBMClassifier(random_state=args["RANDOM_SEED"],
                          num_leaves=50,
                          max_depth=6,
                          learning_rate=0.1,
                          colsample_bytree=0.8)
    xgb = XGBClassifier(random_state=args["RANDOM_SEED"],
                        objective='binary:logistic',eval_metric='logloss',
                        max_depth=3,
                        n_estimators=140)
    bag = BaggingClassifier(random_state=args["RANDOM_SEED"],
                            max_features=1.0,
                            max_samples=0.5,
                            n_estimators=200)
    gbc = GradientBoostingClassifier(random_state=args["RANDOM_SEED"],
                                     learning_rate=0.05,
                                     n_estimators=1000)
    
    model_list = [ran, ext, lgbm, xgb, bag, gbc]
    
    return model_list
