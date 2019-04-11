import tensorflow as tf
import tensorlayer as tl
import numpy as np
import pandas as pd
from statsmodels.tsa.tsatools import lagmat
from sklearn.preprocessing import MinMaxScaler
from analyse import load_train_data, load_test_data, load_data
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error     #均方误差
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--look_back', type=int, default=72, help='look back time: S')
parser.add_argument('--k', type=int, default=12, help=' K ')
parser.add_argument('--look_ahead', type=int, default=48, help='look ahead time')
parser.add_argument('--n_hidden', type=int, default=64, help='LSTM n_hidden units')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--n_epoch', type=int, default=3000, help='n_epoch=500')
parser.add_argument('--print_freq', type=int, default=20, help='train state print freq')
parser.add_argument('--learningRate', type=float, default=0.0001)
parser.add_argument('--data_name', type=str, default='1501', help='data name')

# look_back  k  look_ahead  n_hidden  batch_size  n_epoch  print_freq  learningRate

args = parser.parse_args()
tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

sess = tf.InteractiveSession()

look_back = args.look_back     # S
look_ahead = args.look_ahead
k = args.k              # K
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
trainstd, teststd = load_data(data_name=args.data_name) #scaler.fit_transform(train.values.astype(float).reshape(-1, 1))
trainX, trainY = load_train_data(k, trainstd, timestep=1, look_back=look_back, look_ahead=look_ahead)
testX, testY = load_test_data(k, teststd, timestep=1, look_back=look_back, look_ahead=look_ahead)
train_data_x = scaler_x.fit_transform(trainX)
train_data_y = scaler_y.fit_transform(trainY)
train_data_x = train_data_x.reshape(-1, 1, train_data_x.shape[1])[: -(look_ahead - 1)]   # timestep = 1
trainX, trainY = train_data_x, train_data_y
test_data_x = scaler_x.transform(testX)
testX = test_data_x.reshape(-1, 1, look_back)[: -(look_ahead - 1)]
valX = testX
valY = scaler_y.transform(testY)

# 定义placeholder
x = tf.placeholder(tf.float32, shape=[None, 1, look_back], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, look_ahead], name='y_')

# 建立模型
network = tl.layers.InputLayer(inputs=x, name='input_layer')
network = tl.layers.RNNLayer(network, cell_fn=tf.nn.rnn_cell.LSTMCell, cell_init_args={},
                             n_hidden=args.n_hidden, initializer=tf.random_uniform_initializer(0, 0.05), n_steps=1,
                             return_last=False, return_seq_2d=True, name='lstm_layer')
network = tl.layers.DenseLayer(network, n_units=look_ahead, act=tf.identity, name='output_layer')

# 定义损失函数
y = network.outputs
cost = tl.cost.mean_squared_error(y, y_, is_mean=True)

# 定义优化器
train_param = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=args.learningRate, use_locking=False).minimize(cost, var_list=train_param)

# 初始化参数
tl.layers.initialize_global_variables(sess)
tl.files.load_and_assign_npz(sess=sess, name='S{}_K{}_la{}_nh{}_bs{}_models.npz'
                  .format(args.look_back, args.k, args.look_ahead, args.n_hidden, args.batch_size), network=network)
# 列出模型信息
network.print_layers()
network.print_params()

# 训练模型
tl.utils.fit(sess, network, train_op, cost, trainX, trainY, x, y_,
             batch_size=args.batch_size, n_epoch=args.n_epoch, print_freq=args.print_freq, X_val=valX, y_val=valY,
             eval_train=True)
tl.files.save_npz(network.all_params, name='S{}_K{}_la{}_nh{}_bs{}_models.npz'
                  .format(args.look_back, args.k, args.look_ahead, args.n_hidden, args.batch_size))
# 预测
prediction = tl.utils.predict(sess, network, testX[:1], x, y)
mse = mean_squared_error(scaler_y.transform(testY[:1]), prediction)
rmse = mse**0.5
print('mse is: %f, rmse is %f'%(mse, rmse))
prediction = scaler_y.inverse_transform(prediction).reshape(-1, 1)

# 绘制结果
actual = testY[:1].reshape(-1, 1)
plt.plot(prediction, label='prediction')
plt.plot(actual, label='actual')
plt.legend(loc='best')
plt.title("S%d_K%d_look_ahead%d_n_hidden%d_batch_size%d_MSE: %.3f RMSE: %.3f"%
          (args.look_back, args.k, args.look_ahead, args.n_hidden, args.batch_size,mse, rmse), fontsize=8)                # look_back  k  look_ahead  n_hidden  batch_size  n_epoch  print_freq  learningRate

plt.savefig("S%d_K%d_look_ahead%d_n_hidden%d_batch_size%d_MSE%.3f_RMSE%.3f.jpg"%
          (args.look_back, args.k, args.look_ahead, args.n_hidden, args.batch_size,mse, rmse))
plt.show()
# 保存模型
sess.close()

