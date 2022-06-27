import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


# 엑셀 데이터 로드
def load_data(args):
    """
    :param args: args.py의 하이퍼파라미터
    :return: '스크랩 종류 및 양', '부원료', 'Unnamed: 0', '데이터명'등의 불필요 칼럼은 제거하고, '스크랩 종류 및 량' 시트와 merge한 데이터프레임(df)을 반환합니다.
    """
    df = pd.read_excel(args["EXCEL_PATH"], sheet_name=0, header=7) # rows : 155805 | Heats : 2209(1~4월 데이터 기준)
    df = df.drop(['스크랩 종류 및 양', '부원료', 'Unnamed: 0', '데이터명'], axis=1)
    df.columns = list(df.columns[:4]) + [x.split('\n')[0] for x in df.columns[4:]]
    df['시간'] = df['시간'].apply(lambda x: x.round(freq='T'))

    scrap = pd.read_excel(args["EXCEL_PATH"], sheet_name=1, header=1)
    scrap = scrap.rename(columns={"HEAT": "히트NO."})
    scrap["총사용량"] = scrap["총사용량"] * 0.001 # kg -> t
    scrap = scrap.drop(["Unnamed: 0", "조", "생산량", "일자", "강종", "강종명", "강종구분"], axis=1)
    scrap = scrap.fillna(0.0)
    scrap = scrap.iloc[:, :2]
    
    df = pd.merge(df, scrap, on='히트NO.', how='left')
    
    return df


'''
전처리 전체 코드

    전력사용량 기준 전처리
* 1. 처음 부분[:10]에서 전력 사용량이 20 이상인 히트 찾기. 그 이후 지점부터 시작하도록 잘라내기
* 2. 전력 변화량이 7 이상인 히트 찾기. 그 지점까지만 잘라내기. 자르고 난 후, 데이터 길이기 30 미만인 히트 제거.
* 3. 처음 전력사용량이 10보다 큰 히트는 제거

    EBT_OPEN 전처리
* 4. 초반 0~15분 내 1의 값을 갖는 EBT_OPEN은 모두 0 처리
* 5. 전체 데이터 내 EBT_OPEN=1의 값이 없으면, 마지막 행 prev+1개를 EBT_OPEN=1로 처리 
* 6. EBT_OPEN 값이 하나라도 있으면, 최초 EBT_OPEN 기준으로 전 후를 각각 prev, post분을 1로 설정

    변수 추가(Feature Engineering
* 7. 13개의 새로운 변수 추가

    필요 없는 칼럼 제거
* 8. 실험 결과, 학습에 확실히 도움이 되지 않는 칼럼 제거. 추후 추가적으로 칼럼 제거하도록 설정

'''


def elec_preprocess_total(df, args):
    """
    :param df: 불러온 엑셀 데이터를 입력받습니다.
    :param args: args.py의 하이퍼파라미터
    :return: 전력사용량 및 EBT_OPEN 전처리한 데이터프레임을 반환합니다.
    """
    total = None
    names = df['히트NO.'].unique()
    for idx, name in tqdm(enumerate(names), total=len(names)):
        temp = df[df['히트NO.'] == name].reset_index(drop=True)
        
        # 처음 10행에서 전력사용량의 절댓값이 20보다 작은 경우의 index 값 가져오기
        bool_idx_1 = temp['전력사용량'][:10].apply(lambda x: np.abs(x) < 20)
        
        # 전력사용량이 20보다 큰 경우가 있으면, 20보다 작은 행만 가져오기.
        if sum(bool_idx_1 == False) != 0:
            temp = pd.concat([temp[:10][bool_idx_1], temp[10:]], axis=0).reset_index(drop=True)
        else:
            pass
        
        # 전력변화량 변수 추가
        temp["전력변화량"] = temp['전력사용량']-temp['전력사용량'].shift(1).fillna(0)
        
        # 전력변화량이 7보다 큰 경우 index 값 가져오기
        bool_idx_2 = temp['전력변화량'].apply(lambda x: np.abs(x) > 7)
        
        # bool_idx_2(전력변화량 7보다 큰 경우)가 한개 이상이면, 그 행 이전까지만 데이터 잘라내기
        if sum(bool_idx_2) != 0:
            temp = temp.iloc[:bool_idx_2[bool_idx_2 == True].index[0], :]
            
        # 자르고 난 후, 데이터 길이기 30 미만인 히트면 제거
        if len(temp) < 30:
            continue
        
        # 첫 행의 전력사용량이 10 이하인 히트만 가져오기
        temp = temp.reset_index(drop=True)
        if temp.loc[0, '전력사용량'] > 10:
            continue
        
        # EBT_OPEN 전처리
        temp = ebt_preprocess(temp, prev=args["PREV"], post=args["POST"])
        
        # Feature Engineering
        temp = feature_engineering(temp)
        
        if idx == 0:
            total = temp
        else:
            total = pd.concat([total, temp], axis=0)
    
    # 학습에 사용하지 않을 변수 제거
    total = total.drop(["T.T.T",        # 출강 전까지 알 수 없는 변수입니다.
                        "2차장입 후 시간",  # 시각화 결과, 정확하지 않은 값들이 많았습니다.
                        "로체냉각수유량",    # 대부분의 값이 630이었고, 학습에 큰 도움이 되지 않았습니다.
                        "수냉덕트 유량",    # 모든 값이 1600이었습니다.
                        "세틀링 챔버 유량",  # 분단위로 거의 바뀌지 않는 값이었습니다. 또한 결측치가 많았습니다.(29756개)
                        "백필터 온도",      # 결측치가 많았습니다.(54617개)
                        "출강온도",        # 출강 전까지 알 수 없는 변수입니다.
                        "시간",           # TIME이라는 변수를 따로 추가했습니다.
    ], axis=1)
    
    total = total[~total.isnull().any(axis=1)].reset_index(drop=True) # 결측치 존재하는 행 삭제
    
    return total.reset_index(drop=True)


def ebt_preprocess(temp, prev=3, post=1):
    """
    elec_preprocess_total 함수에서 사용되는 보조 함수입니다.
    :param temp: 히트별 데이터프레임을 입력받습니다.
    :param prev: 최초로 EBT_OPEN=1이 되는 시점 이전 prev 행
    :param post: 최초로 EBT_OPEN=1이 되는 시점 이후 post 행
    :return: EBT_OPEN 전처리된 히트별 데이터프레임을 반환합니다.
    """
    # 초반 0~15분 내 1의 값을 갖는 EBT_OPEN은 모두 0 처리
    temp["EBT_OPEN"][:15]=0
  
    #  1의 값이 없으면, 마지막 행 prev+1개를 EBT_OPEN=1로 처리
    if temp["EBT_OPEN"].sum() == 0:
        # (prev+1)개를 1로 설정
        temp['EBT_OPEN'][-(prev+1):] = 1

    else:
        # 최초 EBT_OPEN 기준으로, 전 후 n분을 1로 설정
        temp = temp.iloc[:temp[temp["EBT_OPEN"].apply(lambda x: x == 1) == True].index[0]+(post+1)]
        temp["EBT_OPEN"][-prev-(post+1):] = 1
    return temp


# 변수 추가
def feature_engineering(temp):
    """
    elec_preprocess_total 함수에서 사용되는 보조 함수입니다.
    :param temp: 히트별 데이터프레임을 입력받습니다.
    :return: 새로운 변수가 추가된 데이터프레임을 반환합니다.
    """
    temp['로체 입출구온도차이'] = temp['로체 출구온도'] - temp['로체 입구온도']                           
    temp['천정 입출구온도차이'] = temp['천정 출구온도'] - temp['천정 입구온도']                            
    temp['수냉덕트 입출구온도차이'] = temp['수냉덕트 출구온도'] - temp['수냉덕트 입구온도']   
    temp['로체입출구온도차이_CUMSUM'] = temp["로체 입출구온도차이"].cumsum()
    temp['천정입출구온도차이_CUMSUM'] = temp["천정 입출구온도차이"].cumsum()
    temp['수냉덕트입출구온도차이_CUMSUM'] = temp["수냉덕트 입출구온도차이"].cumsum()
    temp["천정비율"] = temp["천정 입구온도"] / temp["천정 출구온도"]
    temp["로체비율"] = temp["로체 입구온도"] / temp["로체 출구온도"]
    temp["수냉덕트비율"] = temp["수냉덕트 입구온도"] / temp["수냉덕트 출구온도"]
    temp["공냉덕트입구온도변화량"] = temp['공냉덕트 입구온도']-temp['공냉덕트 입구온도'].shift(1).fillna(0)
    temp['TIME'] = range(1, len(temp)+1)   
    temp["전력사용량_NORMALIZED"] = temp["전력사용량"]/temp["총사용량"]
    temp["TIME_NORMALIZED"] = temp["TIME"]/temp["총사용량"]   

    return temp


# 변수 제거
def drop_columns(total, args):
    """
    :param total: 전체 데이터프레임을 입력받습니다.
    :param args: args.py의 하이퍼파라미터
    :return: args에서 DROP_COLS의 값이 있으면 해당 칼럼을 제거합니다.
    """
    if len(args["DROP_COLS"]) != 0:
        total2 = total.drop(args["DROP_COLS"], axis=1)
        print(f"Dropping {len(args['DROP_COLS'])} columns ...")
    else:
        total2 = total
        print("No dropping columns ...")
    return total2
