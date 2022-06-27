import time


def train(model_list, x_train, y_train):
    """
    :param model_list: 학습이 되지 않은 모델 리스트를 입력받습니다.
    :param x_train: 학습에 사용될 독립변수를 입력받습니다.
    :param y_train: 학습에 사용될 종속변수를 입력받습니다.
    :return: 학습이 완료된 모델을 리스트에 저장하여 반환합니다.
    """
    for ind, mod in enumerate(model_list):
        a = time.time()
        mod.fit(x_train, y_train)
        print(f"{ind + 1}/{len(model_list)} {str(model_list[ind]).split('(')[0]} ... " +
              "Time: {:.1f} ...".format(time.time() - a))
    return model_list
