import tensorflow as tf
import numpy as np
import scipy.io as sio
import Pre_process
print(tf.__version__)

#定义读取mat文件的函数
#注意，这里没有加滤波器，后续可以加上滤波器

def get_data_from_mat(path):
    sub_1 = sio.loadmat(path)
    data = sub_1['data']
    data = Pre_process.butter_bandpass_filtfilt(data)#调用巴特沃斯阶滤波器滤波
    data = Pre_process.min_max_scalar(data)#对每一帧图像进行最大最小标准化
    data = data.reshape(-1,16,8)#（帧数,维度1,维度2）把每一帧的数据转换成图像的格式
    gesture = sub_1['gesture']
    gesture = gesture.reshape(-1,)#将原本为2维的gesture数组转化成一维数组
    return data,gesture






    


