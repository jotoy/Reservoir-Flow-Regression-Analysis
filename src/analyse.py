import csv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.tsatools import lagmat
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler

def test():
    d = pd.read_csv('data/waterdata.csv')

    d1501 = d[
        d['TIME'].str.contains(r'2015/1/') + d['TIME'].str.contains(r'2015/2/') + d['TIME'].str.contains(r'2015/3/')]
    d1502 = d[
        d['TIME'].str.contains(r'2015/4/') + d['TIME'].str.contains(r'2015/5/') + d['TIME'].str.contains(r'2015/6/')]
    d1503 = d[
        d['TIME'].str.contains(r'2015/7/') + d['TIME'].str.contains(r'2015/8/') + d['TIME'].str.contains(r'2015/9/')]
    d1504 = d[
        d['TIME'].str.contains(r'2015/10/') + d['TIME'].str.contains(r'2015/11/') + d['TIME'].str.contains(r'2015/12/')]

    d1601 = d[
        d['TIME'].str.contains(r'2016/1/') | d['TIME'].str.contains(r'2016/2/') + d['TIME'].str.contains(r'2016/3/')]
    d1602 = d[
        d['TIME'].str.contains(r'2016/4/') + d['TIME'].str.contains(r'2016/5/') + d['TIME'].str.contains(r'2016/6/')]
    d1603 = d[
        d['TIME'].str.contains(r'2016/7/') + d['TIME'].str.contains(r'2016/8/') + d['TIME'].str.contains(r'2016/9/')]
    d1604 = d[
        d['TIME'].str.contains(r'2016/10/') + d['TIME'].str.contains(r'2016/11/') + d['TIME'].str.contains(r'2016/12/')]

    d1701 = d[
        d['TIME'].str.contains(r'2017/1/') + d['TIME'].str.contains(r'2017/2/') + d['TIME'].str.contains(r'2017/3/')]
    d1702 = d[
        d['TIME'].str.contains(r'2017/4/') + d['TIME'].str.contains(r'2017/5/') + d['TIME'].str.contains(r'2017/6/')]
    d1703 = d[
        d['TIME'].str.contains(r'2017/7/') + d['TIME'].str.contains(r'2017/8/') + d['TIME'].str.contains(r'2017/9/')]
    d1704 = d[
        d['TIME'].str.contains(r'2017/10/') + d['TIME'].str.contains(r'2017/11/') + d['TIME'].str.contains(r'2017/12/')]

    d1801 = d[
        d['TIME'].str.contains(r'2018/1/') + d['TIME'].str.contains(r'2018/2/') + d['TIME'].str.contains(r'2018/3/')]

    d1501_train, d1501_test = train_test_split(d1501, test_size=0.1, random_state=0, shuffle=False)
    d1502_train, d1502_test = train_test_split(d1502, test_size=0.1, random_state=0, shuffle=False)
    d1503_train, d1503_test = train_test_split(d1503, test_size=0.1, random_state=0, shuffle=False)
    d1504_train, d1504_test = train_test_split(d1504, test_size=0.1, random_state=0, shuffle=False)

    d1601_train, d1601_test = train_test_split(d1601, test_size=0.1, random_state=0, shuffle=False)
    d1602_train, d1602_test = train_test_split(d1602, test_size=0.1, random_state=0, shuffle=False)
    d1603_train, d1603_test = train_test_split(d1603, test_size=0.1, random_state=0, shuffle=False)
    d1604_train, d1604_test = train_test_split(d1604, test_size=0.1, random_state=0, shuffle=False)

    d1701_train, d1701_test = train_test_split(d1701, test_size=0.1, random_state=0, shuffle=False)
    d1702_train, d1702_test = train_test_split(d1702, test_size=0.1, random_state=0, shuffle=False)
    d1703_train, d1703_test = train_test_split(d1703, test_size=0.1, random_state=0, shuffle=False)
    d1704_train, d1704_test = train_test_split(d1704, test_size=0.1, random_state=0, shuffle=False)

    d1801_train, d1801_test = train_test_split(d1801, test_size=0.1, random_state=0, shuffle=False)

    # d1501.to_csv('data/d1501.csv')
    # d1502.to_csv('data/d1502.csv')
    # d1503.to_csv('data/d1503.csv')
    # d1504.to_csv('data/d1504.csv')
    # d1601.to_csv('data/d1601.csv')
    # d1602.to_csv('data/d1602.csv')
    # d1603.to_csv('data/d1603.csv')
    # d1604.to_csv('data/d1604.csv')
    # d1701.to_csv('data/d1701.csv')
    # d1702.to_csv('data/d1702.csv')
    # d1703.to_csv('data/d1703.csv')
    # d1704.to_csv('data/d1704.csv')
    # d1801.to_csv('data/d1801.csv')

    # with open('data/waterdata.csv') as f:
    #     f_csv = csv.reader(f)
    #     headings = next(f_csv)
    #     Row = namedtuple('Row', headings)
    #     for r in f_csv:
    #         row = Row(*r)

    # # Import Dataset
    # df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mtcars.csv")
    #
    # # Plot
    # plt.figure(figsize=(12,10), dpi= 80)
    # sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)
    #
    # # Decorations
    # plt.title('Correlogram of mtcars', fontsize=22)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.show()
    # # Import Data
    # df = pd.read_csv('https://github.com/selva86/datasets/raw/master/AirPassengers.csv')
    # # df = pd.read_csv('data/waterdata.csv')
    #
    # # Draw Plot
    #
    # plt.figure(figsize=(16,10), dpi= 80)
    # plt.plot('date', 'traffic', data=df, color='tab:red')
    #
    # # Decoration
    # plt.ylim(50, 750)
    # xtick_location = df.index.tolist()[::12]
    # xtick_labels = [x[-4:] for x in df.date.tolist()[::12]]
    # plt.xticks(xtick_location, xtick_labels, rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    # plt.yticks(fontsize=12, alpha=.7)
    # plt.title("Air Passengers Traffic (1949 - 1969)", fontsize=22)
    # plt.grid(axis='both', alpha=.3)
    #
    # # Remove borders
    # plt.gca().spines["top"].set_alpha(0.0)
    # plt.gca().spines["bottom"].set_alpha(0.3)
    # plt.gca().spines["right"].set_alpha(0.0)
    # plt.gca().spines["left"].set_alpha(0.3)
    # plt.show()

    df = pd.read_csv('data/d1703.csv')

    # Draw Plot
    plt.figure(figsize=(16, 10), dpi=80)
    plt.plot('TIME', 'xiaoxi_out', data=df, color='tab:red', label='xiaoxi_out')
    plt.plot('TIME', 'zhexi_in', data=df, color='tab:blue', label='zhexi_in')
    plt.plot('TIME', 'lengshuijiang_add', data=df, color='tab:pink', label='leng_add')
    plt.plot('TIME', 'xinhua_add', data=df, color='tab:green', label='xinhua_add')
    plt.plot('TIME', 'zhexi_add', data=df, color='y', label='zhexi_add')
    # Decoration
    # plt.ylim(50, 750)
    xtick_location = df.index.tolist()[::216]
    xtick_labels = [x[:-4] for x in df.TIME.tolist()[::216]]
    plt.xticks(xtick_location, xtick_labels, rotation=45, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title("", fontsize=22)
    plt.grid(axis='both', alpha=.3)

    # Remove borders
    plt.gca().spines["top"].set_alpha(0.0)
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.0)
    plt.gca().spines["left"].set_alpha(0.3)
    plt.legend(loc='upper right')
    plt.show()

def load_data(data_name = '1501'):
    d = pd.read_csv('data/waterdata.csv')

    if data_name == '1501':
        d1501 = d[
        d['TIME'].str.contains(r'2015/1/') | d['TIME'].str.contains(r'2015/2/') | d['TIME'].str.contains(r'2015/3/')]
        train, test = train_test_split(d1501, test_size=0.1, random_state=0, shuffle=False)
        return train, test

    if data_name == '1502':
        d1502 = d[
        d['TIME'].str.contains(r'2015/4/') | d['TIME'].str.contains(r'2015/5/') | d['TIME'].str.contains(r'2015/6/')]
        train, test = train_test_split(d1502, test_size=0.1, random_state=0, shuffle=False)
        return train, test
    if data_name == '1503':
        d1503 = d[
        d['TIME'].str.contains(r'2015/7/') | d['TIME'].str.contains(r'2015/8/') | d['TIME'].str.contains(r'2015/9/')]
        train, test = train_test_split(d1503, test_size=0.1, random_state=0, shuffle=False)
        return train, test
    if data_name == '1504':
        d1504 = d[
        d['TIME'].str.contains(r'2015/10/') | d['TIME'].str.contains(r'2015/11/') | d['TIME'].str.contains(r'2015/12/')]
        train, test = train_test_split(d1504, test_size=0.1, random_state=0, shuffle=False)
        return train, test

    if data_name == '1601':
        d1601 = d[
        d['TIME'].str.contains(r'2016/1/') | d['TIME'].str.contains(r'2016/2/') | d['TIME'].str.contains(r'2016/3/')]
        train, test = train_test_split(d1601, test_size=0.1, random_state=0, shuffle=False)
        return train, test
    if data_name == '1602':
        d1602 = d[
        d['TIME'].str.contains(r'2016/4/') | d['TIME'].str.contains(r'2016/5/') | d['TIME'].str.contains(r'2016/6/')]
        train, test = train_test_split(d1602, test_size=0.1, random_state=0, shuffle=False)
        return train, test
    if data_name == '1603':
        print('loading 1603')
        d1603 = d[
        d['TIME'].str.contains(r'2016/7/') | d['TIME'].str.contains(r'2016/8/') | d['TIME'].str.contains(r'2016/9/')]
        train, test = train_test_split(d1603, test_size=0.1, random_state=0, shuffle=False)
        return train, test
    if data_name == '1604':
        d1604 = d[
        d['TIME'].str.contains(r'2016/10/') | d['TIME'].str.contains(r'2016/11/') | d['TIME'].str.contains(r'2016/12/')]
        train, test = train_test_split(d1604, test_size=0.1, random_state=0, shuffle=False)
        return train, test

    if data_name == '1701':
        d1701 = d[
        d['TIME'].str.contains(r'2017/1/') | d['TIME'].str.contains(r'2017/2/') | d['TIME'].str.contains(r'2017/3/')]
        train, test = train_test_split(d1701, test_size=0.1, random_state=0, shuffle=False)
        return train, test
    if data_name == '1702':
        d1702 = d[
        d['TIME'].str.contains(r'2017/4/') | d['TIME'].str.contains(r'2017/5/') | d['TIME'].str.contains(r'2017/6/')]
        train, test = train_test_split(d1702, test_size=0.1, random_state=0, shuffle=False)
        return train, test
    if data_name == '1703':
        d1703 = d[
        d['TIME'].str.contains(r'2017/7/') | d['TIME'].str.contains(r'2017/8/') | d['TIME'].str.contains(r'2017/9/')]
        train, test = train_test_split(d1703, test_size=0.1, random_state=0, shuffle=False)
        return train, test
    if data_name == '1704':
        d1704 = d[
        d['TIME'].str.contains(r'2017/10/') | d['TIME'].str.contains(r'2017/11/') | d['TIME'].str.contains(r'2017/12/')]
        train, test = train_test_split(d1704, test_size=0.1, random_state=0, shuffle=False)
        return train, test

    if data_name == '1801':
        d1801 = d[
        d['TIME'].str.contains(r'2018/1/') | d['TIME'].str.contains(r'2018/2/') | d['TIME'].str.contains(r'2018/3/')]
        train, test = train_test_split(d1801, test_size=0.1, random_state=0, shuffle=False)
        return train, test

def load_train_data(k, train, timestep=1, look_back=1, look_ahead=1):
    X_time = []
    X = train.xiaoxi_out
    X1 = train.xinhua_add
    X2 = train.lengshuijiang_add
    X3 = train.zhexi_add

    y = train.zhexi_in
    X_train = [0 for i in range(len(X) - k)]
    y_train = [0 for i in range(len(X) - k)]
    for i in range(len(X)-k):

        X_train[i] = (X[X.index[i]] + X1[X1.index[i+k]] + X2[X2.index[i+k]] + X3[X3.index[i+k]])
        y_train[i] = (y[y.index[i]])

    dataX = lagmat(X_train, maxlag=look_back, trim='both', original='ex')
    dataY = lagmat(y_train[look_back:], maxlag=look_ahead, trim='backward', original='ex')

    return np.array(dataX), np.array(dataY[: -(look_ahead - 1)])


def load_test_data(k,test, timestep=1, look_back=1, look_ahead=1):
    X = test.xiaoxi_out
    X1 = test.xinhua_add
    X2 = test.lengshuijiang_add
    X3 = test.zhexi_add

    y = test.zhexi_in
    X_test = [0 for i in range(len(X) - k)]
    y_test = [0 for i in range(len(X) - k)]

    for i in range(len(X) - k):
        X_test[i] = (X[X.index[i]] + X1[X1.index[i + k]] + X2[X2.index[i + k]] + X3[X3.index[i + k]])
        y_test[i] = (y[y.index[i]])

    dataX = lagmat(X_test, maxlag=look_back, trim='both', original='ex')
    dataY = lagmat(y_test[look_back:], maxlag=look_ahead, trim='backward', original='ex')

    return np.array(dataX), np.array(dataY[: -(look_ahead - 1)])
