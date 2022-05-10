import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
def svm_main():
    data = pd.read_csv('newdata.csv')
    x = data.drop(columns='41', axis=1)
    y = data['41']
    # print(y)

    # 数据分割
    # 7比3划分训练集，测试集，设置随机种子，保证实验能够复现
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.3,
                                                        stratify=y,
                                                        random_state=2022)
    ## SVM模型
    from sklearn.svm import SVC
    model_SVM = SVC(C=1.0,
                    kernel='rbf',
                    degree=3,
                    gamma='scale',
                    coef0=0.0,
                    shrinking=True,
                    probability=True,
                    tol=0.001,
                    cache_size=200,
                    class_weight=None,
                    verbose=False,
                    max_iter=-1,
                    decision_function_shape='ovr',
                    break_ties=False,
                    random_state=2022)
    model_SVM.fit(x_train, y_train)
    # 输出模型评价指标
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
    y_pred2 = model_SVM.predict(x_test)

    def roc_curve_and_score(y_test, pred_proba):
        roc_auc = roc_auc_score(y_test.ravel(), pred_proba.ravel())
        return roc_auc

    print('SVM测试集')
    precision = precision_score(y_test, y_pred2, average='weighted')
    recall = recall_score(y_test, y_pred2, average='weighted')
    f1score = f1_score(y_test, y_pred2, average='weighted')
    accuracy = accuracy_score(y_test, y_pred2)
    auc = roc_curve_and_score(y_test, model_SVM.predict_proba(x_test)[:, 1])
    print("AUC:", auc)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("f1_score：", f1score)

    # SVM混淆矩阵
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    plt.figure()
    matrix = confusion_matrix(y_test, y_pred2)
    dataframe = pd.DataFrame(matrix, index=['0', '1'], columns=['0', '1'])
    sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt='.5g', square=True)
    plt.title("SVM Confusion Matrix"), plt.tight_layout()
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")
    plt.savefig("static/assets/img/svm_1.jpg")
    plt.clf()

    # 绘制ROC曲线
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    def roc_curve_and_score(y_test, pred_proba):
        fpr, tpr, _ = roc_curve(y_test.ravel(), pred_proba.ravel())
        roc_auc = roc_auc_score(y_test.ravel(), pred_proba.ravel())
        return fpr, tpr, roc_auc

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 12})
    plt.grid()

    # 与逻辑回归对比
    lr = LogisticRegression(max_iter=10000)
    lr.fit(x_train, y_train)
    y_pred1 = lr.predict(x_test)

    def roc_curve_and_score(y_test, pred_proba):
        roc_auc = roc_auc_score(y_test.ravel(), pred_proba.ravel())
        return roc_auc

    p2 = precision_score(y_test, y_pred1, average='weighted')
    r2 = recall_score(y_test, y_pred1, average='weighted')
    f12 = f1_score(y_test, y_pred1, average='weighted')
    acc2 = accuracy_score(y_test, y_pred1)
    auc2 = roc_curve_and_score(y_test, lr.predict_proba(x_test)[:, 1])
    # print(auc2)

    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    x = np.arange(5)
    y = [auc, accuracy, precision, recall, f1score]

    y1 = [auc2, acc2, p2, r2, f12]
    bar_width = 0.35
    tick_label = ["AUC", "ACC", "P", "R", "F1"]
    plt.bar(x, y, bar_width, align="center", color="gold", label="SVM", alpha=0.5)
    plt.bar(x + bar_width, y1, bar_width, color="darkblue", align="center", label="逻辑回归", alpha=0.5)
    plt.ylabel("SVM与logistic", fontsize=15)
    plt.ylim(0.850, 1)
    plt.xticks(x + bar_width / 2, tick_label)
    plt.legend()
    plt.savefig("static/assets/img/svm_2.jpg")
    plt.clf()

    # roc曲线
    def roc_curve_and_score(y_test, pred_proba):
        fpr, tpr, _ = roc_curve(y_test.ravel(), pred_proba.ravel())
        roc_auc = roc_auc_score(y_test.ravel(), pred_proba.ravel())
        return fpr, tpr, roc_auc

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 12})
    plt.grid()

    fpr, tpr, roc_auc = roc_curve_and_score(y_test,
                                            lr.predict_proba(x_test)[:, 1])
    plt.plot(fpr, tpr, color='y', lw=2,
             label='AUC logistic={0:.4f}'.format(roc_auc))
    fpr, tpr, roc_auc = roc_curve_and_score(y_test,
                                            model_SVM.predict_proba(x_test)[:, 1])
    plt.plot(fpr, tpr, color='m', lw=2,
             label='AUC SVM={0:.4f}'.format(roc_auc))

    plt.plot([0, 1], [0, 1], lw=1, linestyle='--')
    plt.legend()
    plt.title('ROC curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.savefig("static/assets/img/svm_3.jpg")
    plt.clf()