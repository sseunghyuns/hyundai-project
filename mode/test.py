import numpy as np
import pandas as pd


# test
def test(model_list, x_test, y_test, test_df):
    """
    :param model_list: 학습이 완료된 모델 리스트를 입력받습니다.
    :param x_test: 검증에 사용될 독립변수를 입력받습니다.
    :param y_test: 검증에 사용될 종속변수를 입력받습니다.
    :param test_df: test셋 데이터프레임을 입력받습니다.
    :return: test_df에 모델의 예측값이 추가된 데이터프레임을 반환합니다.
    """
    p = []
    for mod in model_list:
        predict_proba = mod.predict_proba(x_test)
        p.append(predict_proba[:, 1])

    result_proba = sum(p) / len(model_list)
    result = np.where(result_proba > 0.5, 1, 0)

    predict_df = pd.DataFrame({"label": y_test.values, "predict": result, "predict_prob": np.round(result_proba, 3)})
    total_test_df = pd.concat([test_df, predict_df], axis=1)

    return total_test_df
