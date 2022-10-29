import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# path = os.path.join('a',"b")
# print(path)
# path = 'D:\PycharmProjects'
# print(os.listdir(path))

# a=[np.random.randn(2,3,3),np.random.randn(2,3,3)]
# print(a[1])
# x = np.array(a)
# print(x.shape)


# a = np.random.randn(5,5)
# print(a)
# plt.figure()
# plt.imshow(a)
# plt.colorbar()
# plt.grid(False)
# plt.show()
#
# fig=plt.figure(figsize=(4,3))
# plt.show()

# model = tf.keras.models()
# model.fit(batch_size=1,epochs=)


# matrix=np.random.rand(4,4)
# print(matrix)


"""


import tensorflow as tf
import seaborn as sn
import pandas as pd


y_true = [1, 0, 2, 3, 3, 1]  # 真实标签
y_pred = [2, 0, 2, 3, 3, 1]  # 预测标签

kind = ['one', 'two', 'three', 'four']  # 类别名称


conf = tf.math.confusion_matrix(y_true, y_pred, num_classes=4)  # 计算混淆矩阵
print(conf.numpy())

conf_numpy = conf.numpy()  # 将 Tensor 转化为 NumPy

conf_df = pd.DataFrame(conf_numpy, index=kind, columns=kind)  # 将矩阵转化为 DataFrame
print('cm',conf_df)
print(type(conf_df))
sn.heatmap(conf_df, annot=True, fmt="d", cmap="BuPu")  # 绘制 heatmap
plt.show()


"""
"""
matrix=np.random.rand(4,4)
print(matrix)
y_predict_label=np.argmax(matrix,axis=1)#返回的每一行的下标
print(y_predict_label)

"""
import numpy as np

a = np.array([1, 1, 3, 4, 5, 6, 7, 8, 9])
a = a - 1
print(a)


# b=a!=1
# print(b)
# print(a[b])
# matrix = np.random.rand(9,9)
# print(matrix)
# for i in range(5):
#     print('/n')
# print(matrix[b])


# print(new_a)  # Prints `[1, 2, 5, 6, 8, 9]`

class MyRMS(tf.keras.metrics.Metric):
    def __init__(self, name="myrms", **kwargs):
        super(MyRMS, self).__init__(name=name, **kwargs)
        self.total = self.add_weight('total', shape=(2,), initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        num = tf.square(y_true - y_pred)
        self.total.assign_add(tf.reduce_sum(num, axis=0))
        den = tf.cast(tf.size(tf.reduce_sum(num, axis=-1)), dtype=tf.float32)
        self.count.assign_add(den)
        print(1)

    def forward(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        num = tf.square(y_true - y_pred)
        self.total.assign_add(tf.reduce_sum(num, axis=0))
        den = tf.cast(tf.size(tf.reduce_sum(num, axis=-1)), dtype=tf.float32)
        self.count.assign_add(den)
        x1 = tf.reshape(self.total, (-1, 2))
        x2 = tf.reshape(self.count, (1, -1))
        x2 = tf.tile(x2,(1,2))
        x3 = x1/x2

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.count.assign(0.0)
        self.total.assign(tf.zeros(9))


        print(1)

    def result(self):
        return tf.sqrt(tf.reshape(self.total, (-1, 9)) / tf.tile(tf.reshape(self.count, (1, -1)), (1, 9)))


y_true = [[0., 1.], [0., 0.]]
y_pred = [[1., 1.], [1., 0.]]

# y_true = np.array([[0., 0, 0], [0., 0, 0]])
# y_pred = np.array([[1., 0, 0], [1., 0, 0]])

m = MyRMS()
m.forward(y_true, y_pred)
m.forward(y_true, y_pred)


# Using 'auto'/'sum_over_batch_size' reduction type.
# mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
#
# print(mse(y_true, y_pred).numpy())
# print(1)


def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))


def custom_relative_root_mean_squared_error(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: 每个自由度的rrmse
    """
    num = tf.math.reduce_mean(tf.square(y_true - y_pred), axis=0)
    den = tf.math.reduce_sum(tf.square(y_pred), axis=0) + tf.keras.backend.epsilon()
    return tf.sqrt(num / den)


def custom_root_mean_squared_error(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: 每个自由度的rmse
    """
    return tf.sqrt(tf.math.reduce_mean(tf.square(y_true - y_pred), axis=0))


class MyMAE(tf.keras.metrics.Metric):
    def __init__(self, name="mymae", **kwargs):
        super(MyMAE, self).__init__(name=name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        t = tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=tf.float32)
            ndim = t.ndim
            weight_ndim = sample_weight.ndim
            t = tf.reduce_mean(t, axis=list(range(weight_ndim, ndim)))
            t = tf.multiply(t, sample_weight)

        t_sum = tf.reduce_sum(t)
        self.total.assign_add(t_sum)
        if sample_weight is not None:
            num = tf.reduce_sum(sample_weight)
        else:
            num = tf.cast(tf.size(t), dtype=tf.float32)
        self.count.assign_add(num)

    def result(self):
        return self.total / self.count


# 需要注意的是：
#
# 1.
# 修改权重值，比如total和count，要用self.total.assign_add(), 不能直接对self.total加减。
# 2.
# 有些方法、函数无法在自定义类中使用，编程的时候需要注意。Tensorflow会有错误提示。
#
# 与MeanAbsoluteError对比：
#
#
# ```python
a = tf.random.uniform([2, 3, 4])
b = tf.random.uniform([2, 3, 4])
# w = tf.random.uniform([2, 3])
m = tf.keras.metrics.MeanAbsoluteError()
m.update_state(a, b)
print(m.get_weights())
print(m.result().numpy())
mae = MyMAE()
mae.forward(a, b)
print(mae.get_weights())
print(mae.result().numpy())
