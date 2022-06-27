import argparse


class Args(object):
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel_path', required=False,
                        default='./data/전기로(1월~4월)rev.xlsx',
                        help='보내주신 엑셀 파일의 경로')
    parser.add_argument('--model_save_path', required=False, default='./model',
                        help='학습된 모델의 저장 경로')
    parser.add_argument('--result_path', required=False, default='./result',
                        help='예측 결과 데이터프레임 저장 경로')
    parser.add_argument('--scaler_path', required=False, default='./scaler',
                        help='학습 시 사용된 스케일러(Train셋 각 칼럼의 평균 및 표준편차) 저장 경로')
    parser.add_argument('--prev', required=False, type=int, default=3)
    parser.add_argument('--post', required=False, type=int, default=1)
    parser.add_argument('--fold_num', required=False, type=int, default=5, help='교차 검증 횟수')
    parser.add_argument('--mode', required=False, default='train_split',
                        help="코드 실행 모드 설정. 'train' : 엑셀 파일 전체를 학습, 'train_split' : 엑셀 파일을 훈련 및 검증, 'test' : 엑셀 파일을 검증") # train_split, test
    parser.add_argument('--test_size', required=False, type=float, default=0.2, help='학습, 검증 데이터셋 나눌 비율')
    parser.add_argument('--random_seed', required=False, type=int, default=42, help='난수 고정')
    parser.add_argument('--drop_cols', required=False, default=[], help='제거할 변수들을 리스트 형태로 입력')
    parse = parser.parse_args()

    params = {
        "EXCEL_PATH": parse.excel_path,
        "MODEL_SAVE_PATH": parse.model_save_path,
        "RESULT_PATH": parse.result_path,
        "SCALER_PATH": parse.scaler_path,
        "PREV": parse.prev,
        "POST": parse.post,
        "FOLD_NUM": parse.fold_num,
        "MODE": parse.mode,
        "TEST_SIZE": parse.test_size,
        "RANDOM_SEED": parse.random_seed,
        "DROP_COLS": parse.drop_cols
    }
