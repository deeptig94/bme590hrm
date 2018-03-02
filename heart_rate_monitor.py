import csv
import pandas as pd
import numpy as np
from scipy.signal import argrelmax


class ECGData:

    def __init__(self, file_name=None, threshold_factor=0, user_input=0):
        self.dataset = []
        self.np_ma_dataset = []
        self.ma_dataset = []
        self.max_voltage_index = 0
        self.np_wavelet = []
        self.wavelet = []
        self.correlated_data = []
        self.threshold = 0
        self.threshold_factor = threshold_factor
        self.beat_voltage = []
        self.peaks_index = []
        self.file_name = file_name
        self.beats = []
        self.time_dataset = []
        self.max_min = []
        self.user_input = user_input
        self.hr = 0

    def read_data(self, file_name):

        """Function takes a csv file and creates a pandas dataframe
        :param file_name: csv file name
        :returns self.dataset: a pandas dataframe with columns 'time' and 'voltage'
        """

        self.dataset = pd.read_csv(file_name)
        self.dataset.columns = ['time', 'voltage']
        return self.dataset

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
        self.ma_dataset = self.ma_dataset - self.ma_dataset['voltage'].mean()
        return self.ma_dataset

    def find_max_voltage_index(self):

        """Function returns index at which the voltage value in the moving average pandas dataframe is maximum
        :param self.ma_dataset: moving average ECG pandas dataframe
        :returns self.max_voltage_index: index at maximum voltage value
        """

        self.max_voltage_index = self.ma_dataset['voltage'].idxmax()
        return self.max_voltage_index

    def get_wavelet(self):

        """Function obtains clipped part of the complete dataset that contains 60 voltage and time values
        :param self.ma_dataset: moving average ECG pandas dataframe
        :returns self.wavelet: clipped portion of the filtered, complete dataset containing 60 values
        """

        self.wavelet = self.ma_dataset.drop(self.ma_dataset.index[:(self.max_voltage_index-30)])
        self.wavelet = self.wavelet.drop(self.ma_dataset.index[(self.max_voltage_index+30):])
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
        self.correlated_data[self.correlated_data < 0] = 0
        return self.correlated_data

    def find_peaks(self, threshold_factor):

        """Function returns voltage at each beat
        :param self.correlated_data: numpy 1D array containing correlated and positive data
        :param threshold_factor: factor entered by user to vary maxima cut-off values
        :returns self.beat_voltage: array with voltages at beat
        """

        self.threshold = (np.mean(self.correlated_data))*threshold_factor
        print(self.threshold)
        relative_maxima = argrelmax(self.correlated_data, order=30)
        relative_maxima = np.ravel(relative_maxima)
        self.beat_voltage = self.correlated_data[relative_maxima]
        self.beat_voltage = [i for i in self.beat_voltage if i >= self.threshold]
        return self.beat_voltage

    def find_peaks_index(self):

        """Function returns index for each beat voltage
        :param self.beat_voltage: array with voltages at beats
        :param self.correlated_data: numpy 1D array containing correlated and positive data
        """

        for i, j, in enumerate(self.beat_voltage):
            for p, q in enumerate(self.correlated_data):
                if q == self.beat_voltage[i]:
                    self.peaks_index.append(p)
        return self.peaks_index

    def beats_times(self):

        """Function returns a numpy array of the times at which a beat occurred
        :param self.dataset: original pandas dataframe of raw data
        :param self.peaks_index: index points at which beats occur
        :returns self.beats: times at which beats occur"""

        self.time_dataset = self.dataset.drop(['voltage'], axis=1)
        self.beats = self.time_dataset.values[self.peaks_index]
        self.beats = np.ravel(self.beats)
        print('These are the times when the beats occurred:', self.beats)
        return self.beats

    def num_beats(self):

        """Function returns the number of beats that occurred
        :param self.beat_voltage: numpy array of voltage at every beat that occurred
        :returns len(self.beat_voltage): length of numpy array
        """

        print('This many beats occurred:', len(self.beat_voltage))
        return len(self.beat_voltage)

    def duration(self):

        """Function returns time of ECG
        :param self.time_dataset: pandas dataframe of voltage at every beat that occurred
        :returns max(self.time_dataset.values): time duration of ECG
        """

        print('This is the time duration of the ECG:', max(self.time_dataset.values))
        return max(self.time_dataset.values)

    def voltage_extremes(self):

        """Function returns maximum and minimum voltage values in ECG
        :param self.dataset: pandas dataframe with all raw ECG data
        :returns self.max_min: tuple containing maximum and minimum voltage values"""

        self.max_min = [self.dataset['voltage'].max(), self.dataset['voltage'].min()]
        print('The maximum and minimum voltage values are:', self.max_min)
        return self.max_min

    def mean_hr_bpm(self, user_input):

        """Function returns average heart rate over user specified input
        :param self.time_dataset: pandas dataframe with ECG time data
        :param self.beat_voltage: numpy array of voltage at every beat that occurred
        :returns self.hr: average heart rate in beats per second"""

        if user_input*60 > max(self.time_dataset.values):
            self.hr = len(self.beat_voltage) / max(self.time_dataset.values)
        else:
            self.hr = len(self.beat_voltage)/(user_input*60)

        print('Average heart rate in beats/second is:', self.hr)
        return self.hr



x = ECGData()
x.read_data('test_data/test_data1.csv')
x.moving_average()
x.find_max_voltage_index()
x.get_wavelet()
x.pandas_to_numpy()
x.correlate_wavelet_dataset()
x.find_peaks(7)
x.find_peaks_index()
x.beats_times()
x.num_beats()
x.duration()
x.voltage_extremes()
x.mean_hr_bpm(user_input=2)
