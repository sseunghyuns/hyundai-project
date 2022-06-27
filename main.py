from utils.preprocess import load_data, elec_preprocess_total, drop_columns
from utils.model_utils import data_split, get_train_test_data, get_models, get_train_data, get_test_data
from utils.model_save_load import save_model, load_trained_model
from utils.metrics import model_metrics
from utils.args import Args
from mode.test import test
from mode.train import train
import os


if __name__ == "__main__":
    args = Args().params
    print("Data Loading ...")
    df = load_data(args)
    total = elec_preprocess_total(df, args)
    total2 = drop_columns(total, args)
    print("Data Loaded ...")

    if args["MODE"] == 'train_split':
        """
        엑셀 파일을 8:2로 나누어 학습 및 검증에 사용합니다.
        학습에 사용된 scaler 및 학습 완료된 모델들을 저장합니다.
        예측 결과가 저장된 데이터프레임을 "RESULT_PATH"에 "result.csv' 파일명으로 저장합니다.
        """
        train_df, test_df = data_split(total2, args)
        x_train, y_train, x_test, y_test = get_train_test_data(train_df, test_df, args)
        model_list = get_models(args)
        print(f"X_train : {x_train.shape}\nx_test : {x_test.shape}")
        print("Training Starts ...")
        model_list = train(model_list, x_train, y_train)
        save_model(model_list, args)
        print("Training finished ...")

        total_test_df = test(model_list, x_test, y_test, test_df)

        if not os.path.isdir(args["RESULT_PATH"]):
            os.mkdir(args["RESULT_PATH"])

        total_test_df.to_csv(args["RESULT_PATH"] + "/result.csv", index=False)
        model_metrics(total_test_df)
        print("Prediction finished ...")

    elif args["MODE"] == 'train':
        """
        엑셀 파일 전체를 학습에 사용합니다.
        학습에 사용된 scaler 및 학습 완료된 모델들을 저장합니다.
        """
        x_train, y_train = get_train_data(total2, args)
        model_list = get_models(args)
        print("Training Starts ...")
        model_list = train(model_list, x_train, y_train)
        save_model(model_list, args)
        print("Training finished ...")

    else: # 'test'
        """
        엑셀 파일 전체를 검증합니다.
        미리 저장된 scaler 및 학습된 모델들을 불러와 검증(예측)합니다.
        예측 결과가 저장된 데이터프레임을 "RESULT_PATH"에 "result.csv' 파일명으로 저장합니다.
        """
        model_list = load_trained_model(args)
        x_test, y_test = get_test_data(total2, args)
        total_test_df = test(model_list, x_test, y_test, total2)

        if not os.path.isdir(args["RESULT_PATH"]):
            os.mkdir(args["RESULT_PATH"])

        total_test_df.to_csv(args["RESULT_PATH"] + "/result.csv", index=False)
        model_metrics(total_test_df)
        print("Prediction finished ...")

