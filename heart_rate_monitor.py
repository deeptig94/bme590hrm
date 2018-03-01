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
        #self.convert_file_name = convert_file_name
        #self.voltage = voltage
        #self.time = time

    def read_data(self, file_name):

        """Function takes a csv file and creates a pandas dataframe
        :param file_name: csv file name
        :returns self.dataset: a pandas dataframe with columns 'time' and 'voltage'
        """

        self.dataset = pd.read_csv(file_name)
        self.dataset.columns = ['time', 'voltage']
        #print(self.dataset)
        return self.dataset

    def plot_data(self):

        """Function takes pandas dataframe and plots voltage vs. time
        :param self.dataset: pandas dataframe
        :returns
        """

        plt.title("Heart Rate Signal")
        self.dataset.plot(x='time', y='voltage')
        plt.show()

    def moving_average(self):

        """Function takes pandas dataframe with raw ECG information and performs a moving average function over it
        with a window of 10
        :param self.dataset: pandas dataframe with raw ECG information
        :returns self.ma_dataset: pandas dataframe after moving average is performed
        """

        self.ma_dataset = self.dataset.rolling(window=10, center=False).mean()
        self.ma_dataset = self.ma_dataset.drop(self.ma_dataset.index[:10])
        self.ma_dataset = self.ma_dataset.reset_index()
        del self.ma_dataset['index']
        self.ma_dataset.plot(x='time', y='voltage')
        plt.show()
        #print(self.ma_dataset)
        return self.ma_dataset

    def find_max_voltage_index(self):

        """Function returns index at which the voltage value in the moving average pandas dataframe is maximum
        :param self.ma_dataset: moving average ECG pandas dataframe
        :returns self.max_voltage_index: index at maximum voltage value
        """

        self.max_voltage_index = self.ma_dataset['voltage'].idxmax()
        #print(self.max_voltage_index)
        return self.max_voltage_index

    def get_wavelet(self):

        """Function obtains clipped part of the complete dataset that contains 60 voltage and time values
        :param self.ma_dataset: moving average ECG pandas dataframe
        :returns self.wavelet: clipped portion of the filtered, complete dataset containing 60 values
        """

        self.wavelet = self.ma_dataset.drop(self.ma_dataset.index[:(self.max_voltage_index-30)])
        self.wavelet = self.wavelet.drop(self.ma_dataset.index[(self.max_voltage_index+30):])
        #print(self.wavelet)
        #self.wavelet.plot(x='time', y='voltage')
        #plt.show()
        return self.wavelet

    def pandas_to_numpy(self):

        """Function converts pandas dataframe to numpy array with only voltage values
        :param self.ma_dataset: moving average ECG pandas dataframe
        :param self.wavelet: clipped portion of the filtered, complete dataset containing 60 values
        :returns self.np_ma_dataset: numpy 1D array with all voltage values
        :returns self.np_wavelet: numpy 1D array with clipped portion of voltage values
        """

        self.np_ma_dataset = self.ma_dataset.drop(['time'], axis=1)
        self.np_ma_dataset = self.np_ma_dataset.values
        self.np_ma_dataset = np.ravel(self.np_ma_dataset)
        self.np_wavelet = self.wavelet.drop(['time'], axis=1)
        self.np_wavelet = self.np_wavelet.values
        self.np_wavelet = np.ravel(self.np_wavelet)
        return self.np_ma_dataset, self.np_wavelet

    def correlate_wavelet_dataset(self):

        """Function correlates wavelet with complete set of filtered data
        :param self.np_wavelet: numpy 1D array with clipped portion of voltage values
        :param self.np_ma_dataset: numpy 1D array with all voltage values
        :returns self.correlted_data: numpy 1D array containing correlation data
        """

        self.correlated_data = np.correlate(self.np_wavelet, self.np_ma_dataset)
        self.correlated_data = self.correlated_data - np.mean(self.correlated_data)
        plt.plot(self.correlated_data)
        plt.show()
        print(self.correlated_data)
        return self.correlated_data


x = ECGData('test_data/test_data1.csv')
x.read_data('test_data/test_data1.csv')
#x.plot_data()
x.moving_average()
x.find_max_voltage_index()
x.get_wavelet()
x.pandas_to_numpy()
x.correlate_wavelet_dataset()
#x.auto_correlate()