import pickle
import os


# 학습한 모델 저장
def save_model(model_list, args):
    """
    :param model_list: 학습이 완료된 모델 리스트를 입력받습니다.
    :param args: args.py의 하이퍼파라미터
    :return: 따로 반환값은 없고, 함수 실행 시 학습 완료된 모델을 'MODEL_SAVE_PATH'에 'ensemble_model.pkl' 파일명으로 저장합니다.
    """
    filename = args["MODEL_SAVE_PATH"] + "/ensemble_model.pkl"
    if not os.path.isdir(args["MODEL_SAVE_PATH"]):
        os.mkdir(args["MODEL_SAVE_PATH"])    
    pickle.dump(model_list, open(filename, 'wb')) 
    print("Model saved ...")
    return


# 저장된 모델 불러오기
def load_trained_model(args):
    """
    :param args: args.py의 하이퍼파라미터
    :return: 'MODEL_SAVE_PATH'에 'ensemble_model.pkl'로 저장된 모델들을 불러옵니다. 불러온 모델을 loaded_model_list 변수에 저장하여 반환합니다.
    """
    loaded_model_list = pickle.load(open(args["MODEL_SAVE_PATH"]+"/ensemble_model.pkl", 'rb'))
    return loaded_model_list
