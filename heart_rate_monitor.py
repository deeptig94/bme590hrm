import csv
import pandas as pd
import numpy as np
from scipy.signal import argrelmax, wiener
import matplotlib.pyplot as plt


class ECGData:

    def __init__(self, file_name=None, threshold_factor=0, user_input=0):
        self.dataset = []
        self.pd_dataset = []
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
        self.voltage_dataset = []

    def read_data(self, file_name):

        """Function takes a csv file and creates a numpy array with ECG data
        :param file_name: csv file name
        :returns self.dataset: a numpy array with ECG data
        """

        self.dataset = pd.read_csv(file_name)
        self.pd_dataset = self.dataset
        self.dataset = self.dataset.values
        self.voltage_dataset = np.delete(self.dataset, np.s_[0], axis=1)
        self.time_dataset = np.delete(self.dataset, 1, axis=1)
        return self.dataset

    def moving_average(self):

        """Function takes numpy array with raw ECG information and performs a moving average function over it
        with a window of 10
        :param self.dataset: numpy array with raw ECG information
        :returns self.ma_dataset: numpy array after moving average is performed
        """

        self.voltage_dataset = self.voltage_dataset
        self.ma_dataset = wiener(self.voltage_dataset)
        plt.plot(self.ma_dataset)
        plt.show()
        print(self.ma_dataset)
        return self.ma_dataset

    def find_max_voltage_index(self):

        """Function returns index at which the voltage value in the ECG numpy array is maximum
        :param self.ma_dataset: moving average ECG numpy array
        :returns self.max_voltage_index: index at maximum voltage value
        """
        self.dataset = self.dataset
        self.max_voltage_index = argrelmax(self.dataset, order=len(self.dataset))
        self.max_voltage_index = self.max_voltage_index[0]
        self.max_voltage_index.astype(int)
        print(self.max_voltage_index)
        return self.max_voltage_index

    def get_wavelet(self):

        """Function obtains clipped part of the complete dataset that contains 60 voltage and time values
        :param self.ma_dataset: moving average ECG numpy array
        :returns self.wavelet: clipped portion of the filtered, complete dataset containing 60 values
        """

        self.ma_dataset = self.ma_dataset
        self.wavelet = np.delete(self.ma_dataset, np.s_[:(self.max_voltage_index-30)], axis=1)
        self.wavelet = np.delete(self.ma_dataset, np.s_[(self.max_voltage_index+30):], axis=1)
        plt.plot(self.wavelet)
        plt.show()
        return self.wavelet

    def correlate_wavelet_dataset(self):

        """Function correlates wavelet with complete set of filtered data
        :param self.np_wavelet: numpy 1D array with clipped portion of voltage values
        :param self.np_ma_dataset: numpy 1D array with all voltage values
        :returns self.correlted_data: numpy 1D array containing correlation data
        """

        self.correlated_data = np.correlate(self.wavelet, self.ma_dataset)
        self.correlated_data = self.correlated_data - np.mean(self.correlated_data)
        self.correlated_data[self.correlated_data < 0] = 0
        return self.correlated_data

    def find_peaks(self, threshold_factor):

        """Function returns voltage at each beat
        :param self.correlated_data: numpy 1D array containing correlated and positive data
        :param threshold_factor: factor entered by user to vary maxima cut-off values
        :returns self.beat_voltage: array with voltages at beat
        """

        self.correlated_data = self.correlated_data
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
        :param self.dataset: original numpy array of raw data
        :param self.peaks_index: index points at which beats occur
        :returns self.beats: times at which beats occur"""

        self.peaks_index = self.peaks_index
        self.beats = self.time_dataset[self.peaks_index]
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
        :param self.time_dataset: numpy array of voltage at every beat that occurred
        :returns max(self.time_dataset.values): time duration of ECG
        """

        print('This is the time duration of the ECG:', max(self.time_dataset))
        return max(self.time_dataset)

    def voltage_extremes(self):

        """Function returns maximum and minimum voltage values in ECG
        :param self.dataset: numpy array with all raw ECG data
        :returns self.max_min: tuple containing maximum and minimum voltage values"""

        self.max_min = [self.voltage_dataset.max(), self.voltage_dataset.min()]
        print('The maximum and minimum voltage values are:', self.max_min)
        return self.max_min

    def mean_hr_bpm(self, user_input):

        """Function returns average heart rate over user specified input
        :param self.time_dataset: numpy array with ECG time data
        :param self.beat_voltage:
        :returns self.hr: average heart rate in beats per second"""

        if user_input*60 > max(self.time_dataset):
            self.hr = len(self.beat_voltage) / max(self.time_dataset)
        else:
            self.hr = len(self.beat_voltage)/(user_input*60)

        print('Average heart rate in beats/second is:', self.hr)
        return self.hr

    
def main():
    x = ECGData()
    x.read_data('test_data/test_data1.csv')
    x.moving_average()
    x.find_max_voltage_index()
    x.get_wavelet()
    x.correlate_wavelet_dataset()
    x.find_peaks(7)
    x.find_peaks_index()
    x.beats_times()
    x.num_beats()
    x.duration()
    x.voltage_extremes()
    x.mean_hr_bpm(user_input=2)
