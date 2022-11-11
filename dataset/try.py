from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC

# 下载数据
digits = datasets.load_digits()
# print(digits.images)
n_sample = len(digits.images)
# 把数据转换为二维数据，x的行数据是不同样本数据，列是样本属性。
x = digits.images.reshape(n_sample, -1)  # 取数据的所有行第一列数据
y = digits.target
# print(x)
# 以下方法确定解释变量只能有一个，但是多个解释变量该怎么处理呢,答案是x包含了众多解释变量
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
scores = ['precision', 'recall']
for score in scores:
    print('Tuning hyper-parameters for %s' % score)
    print()
    # 利用网格搜索算法构建评估器模型，并且对数据进行评估
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='%s_macro' % score)
    clf.fit(x_train, y_train)
    print('最优参数：', clf.best_params_)
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print('网格数据得分：', '%0.3f (+/-%0.3f) for %r' % (mean, std, params))
        # 这个std有的文章乘以2，但个人不知道为什么需要乘以2，如有明白的朋友，求指点。
    print()
    y_true, y_pred = y_test, clf.predict(x_test)
    print(y_true)
    print(classification_report(y_true, y_pred))

# 在获取最优超参数之后， 用5折交叉验证来评估模型
clf = SVC(kernel='rbf', C=1, gamma=1e-3)  # 最优模型
# 对模型进行评分
scores = cross_val_score(clf, x, y, cv=5)
print(scores)
