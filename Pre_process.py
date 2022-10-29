import tensorflow as tf
import numpy as np
import scipy.io as sio
from scipy import signal
from PIL import Image
from sklearn.preprocessing import MinMaxScaler


#定义巴特沃斯滤波器,输入的data是时域的函数
#axis=0表示跨行，axis=1表示跨列，作为方法动作的副词

def butter_bandpass_filtfilt(data,cutoff=[20.0,380.0],fs=1000.0,oder=4):
    wn = [2*i/fs for i in cutoff]
    b,a = signal.butter(oder,wn,'bandpass',analog=False)#滤波器构造函数
    output = signal.filtfilt(b,a,data,axis=0)#进行滤波
    return output

#进行归一化处理
def min_max_scalar(data):
    data = np.transpose(data)#因为在归一化的时候只能对一列进行归一化，因此我们先求一下转置
    scaler = MinMaxScaler()  # 实例化
    scaler = scaler.fit(data)  # fit，在这里本质是生成min(x)和max(x)
    result = scaler.transform(data)  # 通过接口导出结果
    result = np.transpose(result)
    return result

"""
测试代码
path_road ='./dataset/001-00'+str(1)+'.mat'
sub_1 = sio.loadmat(path_road)
data = sub_1['data']
data1 = min_max_scalar(data)
data1 = data1.reshape(-1,16,8)
print(data1.shape)
print(data1[0])
gray_image = data1[0]*255
print(gray_image)
picture = Image.fromarray(gray_image)
picture.show()

"""


