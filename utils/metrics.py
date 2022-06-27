import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score


def model_metrics(total_test_df):
    """
    :param total_test_df: 모델로 예측한 결과가 저장된 데이터프레임을 입력받습니다.
    :return: 함수 실행 시, 예측 결과(Precision, Recall, Accuracy, AUC)를 출력하고, Confusion Metrics, ROC 곡선, Precision/Recall 곡선을 시각화합니다.
    """
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(20, 5)

    predict = total_test_df["predict"].values
    label = total_test_df["label"].values
    predict_proba = total_test_df["predict_prob"].values

    # Precisions/Recalls
    precisions, recalls, thresholds = precision_recall_curve(label, predict_proba)

    min_idx = np.argmin(abs(recalls - precisions))
    intersect_point = thresholds[min_idx]

    axes[1].plot(thresholds, precisions[:-1], "b--", label="Precision")
    axes[1].plot(thresholds, recalls[:-1], "g--", label="Recall")
    axes[1].plot(intersect_point, precisions[min_idx], 'xr', label='Intersection point', color='red')
    axes[1].set_title('Precision/Recall Curve')
    axes[1].legend(loc='lower center', prop={'size': 10})
    axes[1].annotate("{:.3f}".format(intersect_point),
                     xy=(intersect_point, precisions[min_idx]),
                     xytext=(intersect_point, precisions[min_idx] + 0.15),
                     ha="center",
                     arrowprops=dict(facecolor='black', shrink=0.15),
                     fontsize=15,
                     )

    # ROC curve
    false_positive_rate, true_positive_rate, threshold = roc_curve(label, predict_proba)
    axes[2].set_title('ROC curve')
    axes[2].plot(false_positive_rate, true_positive_rate)
    axes[2].plot([0, 1], ls="--")
    axes[2].plot([0, 0], [1, 0], c=".7"), axes[2].plot([1, 1], c=".7")
    axes[2].set_ylabel('True Positive Rate')
    axes[2].set_xlabel('False Positive Rate')

    # Confusion Matrix
    print("*" * 50)
    cf_mtx = confusion_matrix(label, predict)
    print(f"Confusion Matrix :\n{cf_mtx}")
    df_cm = pd.DataFrame(cf_mtx, index=[i for i in [0, 1]],
                         columns=[i for i in [0, 1]])
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.viridis, fmt='g', ax=axes[0])

    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")
    axes[0].set_title("Confusion Matrix")

    print("\nAccuracy score: {:.4f}".format(accuracy_score(label, predict)))
    print("Precision score: {:.4f}".format(precision_score(label, predict)))
    print("Recall score: {:.4f}".format(recall_score(label, predict)))
    print("F1 score : {:.4f}".format(f1_score(label, predict)))
    print("AUC: Area Under Curve: {:.4f}\n".format(roc_auc_score(label, predict_proba)))

    print(f"Intersection point : {intersect_point}")
    print("*" * 50)

    plt.show()

