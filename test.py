import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply,
    Permute, Flatten, RepeatVector, Lambda, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
print(tf.__version__)  # 查看tensorflow版本
print(tf.__path__)     # 查看tensorflow安装路径

a = tf.test.is_built_with_cuda()  # 判断CUDA是否可以用
b = tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)  # 判断GPU是否可以用

print(a) # 显示True表示CUDA可用
print(b) # 显示True表示GPU可用

# 查看驱动名称
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print(tf.config.list_physical_devices('GPU'))