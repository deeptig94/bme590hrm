import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import autocorrelation_plot


class ECGData:

    def __init__(self, file_name):
        self.dataset = 0
        self.np_ma_dataset = 0
        self.ma_dataset = 0
        self.max_voltage_index = 0
        self.np_wavelet = 0
        self.wavelet = 0
        self.correlated_data = 0
        self.file_name = file_name
        #self.voltage = voltage
        #self.time = time

    def read_data(self, file_name):
        self.dataset = pd.read_csv(file_name)
        self.dataset.columns = ['time', 'voltage']
        #print(self.dataset)
        return self.dataset

    def plot_data(self):
        plt.title("Heart Rate Signal")
        self.dataset.plot(x='time', y='voltage')
        plt.show()

    def moving_average(self):
        self.ma_dataset = self.dataset.rolling(window=10, center=False).mean()
        self.ma_dataset = self.ma_dataset.drop(self.ma_dataset.index[:10])
        self.ma_dataset = self.ma_dataset.reset_index()
        del self.ma_dataset['index']
        #self.ma_dataset.plot(x='time', y='voltage')
        #plt.show()
        print(self.ma_dataset)
        return self.ma_dataset

    def find_max_voltage_index(self):
        self.max_voltage_index = self.ma_dataset['voltage'].idxmax()
        print(self.max_voltage_index)
        return self.max_voltage_index

    def get_wavelet(self):
        self.wavelet = self.ma_dataset.drop(self.ma_dataset.index[:(self.max_voltage_index-30)])
        self.wavelet = self.wavelet.drop(self.ma_dataset.index[(self.max_voltage_index+30):])
        print(self.wavelet)
        #self.wavelet.plot(x='time', y='voltage')
        #plt.show()
        return self.wavelet

    def pandas_to_numpy(self):
        self.np_ma_dataset = self.ma_dataset.values
        self.np_wavelet = self.wavelet.values
        return self.np_ma_dataset & self.np_wavelet

    def correlate_wavelet_dataset(self):
        self.correlated_data = np.correlate(self.np_wavelet, self.np_ma_dataset)
        self.correlated_data.plot(x='time', y='voltage')
        plt.show()
        print(self.correlated_data)
        return self.correlated_data

    '''def auto_correlate(self):
        #print(dataset['voltage'].autocorr(lag=1))
        autocorrelation_plot(self.dataset)
        plt.show()'''


x = ECGData('test_data/test_data1.csv')
x.read_data('test_data/test_data1.csv')
#x.plot_data()
x.moving_average()
x.find_max_voltage_index()
x.get_wavelet()
x.pandas_to_numpy()
x.correlate_wavelet_dataset()
#x.auto_correlate()