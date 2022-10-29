import tensorflow as tf
import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
import seaborn as sn
import get_data
import pandas as pd
import Pre_process
print(tf.__version__)
"""
Caution: 该算法对每一帧的图像进行train以及test，因此没有使用滑窗分割。
在训练时，每一帧都是有标签的
通过采用多数投票在时域上建立联系 
"""

#连续读取一个文件夹内的数据
raw_data = []#定义存放train_data的空列表
raw_label = []#定义存放train_label的空列表
for i in range(1,7):
    path_road ='./dataset/001-00'+str(i)+'.mat'
    data, gesture = get_data.get_data_from_mat('./dataset/001-00' + str(i) + '.mat')
    raw_data.append(data)
    raw_label.append(gesture)
print('come on')
print(len(raw_data))
print(len(raw_label))


#将列表的数组按照第0维进行合成
temper_array1 = raw_data[0]
temper_array2 = raw_label[0]
for i in range(1,6):
    temper_array1 = np.concatenate((temper_array1, raw_data[i]), axis=0)
    temper_array2 = np.concatenate((temper_array2, raw_label[i]), axis=-1)

#对数据进行维度的扩张
temper_array1 = temper_array1.reshape(temper_array1.shape[0],temper_array1.shape[1],temper_array1.shape[2],1)
print(temper_array1.shape)

#因为标签0的比例太高，导致最后的训练结果太差，删除所有为0的标签
del_index = temper_array2!=0
temper_array2 = temper_array2[del_index]
temper_array2 = temper_array2-1
temper_array1 = temper_array1[del_index]



#划分训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    temper_array1, temper_array2, test_size=0.3, shuffle=True)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#zero_number = np.sum(y_test==0)

#搭建CNN模型注意可能会存在梯度爆炸和梯度消失的问题
net=tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16,kernel_size=4,activation='sigmoid',padding='same',input_shape=(16,8,1)),
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
    #tf.keras.layers.Conv2D(filters=16,kernel_size=2,activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(120,activation='relu'),
    #tf.keras.layers.Dense(84,activation='relu'),
    tf.keras.layers.Dense(6,activation='softmax')])


#编译训练模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)#用SGD优化器优化模型
net.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = net.fit(X_train, y_train, epochs=1, validation_split=0.3)
history_dict = history.history
print(history_dict.keys())

#结果分析
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
epochs = range(1,len(acc)+1)

plt.figure()
plt.plot(epochs,acc,'bo', label = 'Training Acc')
plt.plot(epochs,val_acc,'b', label = 'validation Acc')

plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



#在测试集上预测数据
net.evaluate(X_test, y_test, verbose=1)
#预测
y_predict=net.predict(X_test)
print(y_predict)
y_predict_label=np.argmax(y_predict,axis=1)

#绘制混淆矩阵
conf = tf.math.confusion_matrix(y_test, y_predict_label, num_classes=6)  # 计算混淆矩阵
print(conf.numpy())

conf_numpy = conf.numpy()  # 将 Tensor 转化为 NumPy

# conf_df = pd.DataFrame(conf_numpy, index=kind, columns=kind)  # 将矩阵转化为 DataFrame
# print('cm',conf_df)
# print(type(conf_df))
sn.heatmap(conf_numpy, annot=True, fmt="d", cmap="BuPu")  # 绘制 heatmap
plt.show()


