from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
from decimal import *

font_1 = {'fontproperties': 'Times New Roman', 'color': 'white', 'size': 10}
font_2 = {'fontproperties': 'Times New Roman', 'color': 'black', 'size': 10}
font_3 = {'fontproperties': 'Times New Roman', 'color': 'black', 'size': 20}
#plt.text(x,y,'text',fontdict=font)


def cm_plot(y, yp, gestures):
    labels_name = [str(i) for i in range(gestures)]
    plt.rcParams['figure.figsize'] = (9, 6)
    # plt.rc('font',family='serif',size='10')   # 设置字体样式、大小
    # 按行进行归一化
    cm = metrics.confusion_matrix(y, yp)  # 输出为混淆矩阵
    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.matshow(cm, cmap=plt.cm.Blues)  # 画混淆矩阵图，配色风格使用cm.Greens
    plt.colorbar()  # 颜色标签
    num_local = np.array(range(gestures))
    for x in range(len(cm)):
        for y in range(len(cm)):
            if x == y:
                plt.text(x, y, str(Decimal(cm[y, x]*100).quantize(Decimal('0.0'))), horizontalalignment='center',
                         verticalalignment='center', fontdict=font_1)
            else:
                plt.text(x, y, str(Decimal(cm[y, x]*100).quantize(Decimal('0.0'))), horizontalalignment='center',
                         verticalalignment='center', fontdict=font_2)
    plt.title("Confusion Matrix", fontdict=font_3)
    plt.plot(x, y)
    plt.yticks(num_local, labels_name, fontproperties='Times New Roman', size=15, rotation=0)  # 将标签印在y轴坐标上
    plt.xticks(num_local, labels_name, fontproperties='Times New Roman', size=15, rotation=0)  # 将标签印在x轴坐标上
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('Predicted label', fontdict=font_3)  # 坐标轴标签
    plt.ylabel('True label', fontdict=font_3)  # 坐标轴标签
    return plt
