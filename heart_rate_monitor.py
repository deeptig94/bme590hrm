import json
import pandas as pd
import numpy as np
from scipy.signal import argrelmax, wiener
import logging

logging.basicConfig(filename='logging.txt', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y &I:%M:%S %p',
                    level=logging.DEBUG)


class ECGData:

    def __init__(self, file_name, threshold_factor=0, user_input=0):
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
        self.max_min = []
        self.user_input = user_input
        self.hr = 0
        data = pd.read_csv(self.file_name)
        self.data = data
        dataset = self.data.values
        voltage_dataset = np.delete(dataset, np.s_[0], axis=1)
        time_dataset = np.delete(dataset, 1, axis=1)
        self.voltage_dataset = voltage_dataset
        self.time_dataset = time_dataset
        self.dataset = dataset
        self.dur = 0
        self.beat_count = 0

    def moving_average(self):

        """Function takes numpy array with raw ECG information and performs a moving average function over it
        with a window of 10
        :returns self.ma_dataset: numpy array after the moving average is performed
        """
        self.ma_dataset = wiener(np.asarray(self.voltage_dataset))

        logging.info('INFO: Obtained dataset')
        return self.ma_dataset

    def find_max_voltage_index(self):

        """Function returns index at which the voltage value in the ECG numpy array is maximum
        :returns self.max_voltage_index: index at maximum voltage value
        """
        self.max_voltage_index = argrelmax(np.asarray(self.dataset), order=len(self.dataset))
        self.max_voltage_index = self.max_voltage_index[0]
        return self.max_voltage_index

    def get_wavelet(self):

        """Function obtains clipped part of the complete dataset that contains 60 voltage and time values
        :returns self.wavelet: clipped portion of the filtered, complete dataset containing 60 values
        """
        roi = slice(int(self.max_voltage_index)-30, int(self.max_voltage_index)+30)
        self.wavelet = self.ma_dataset[roi]
        self.wavelet = np.ravel(self.wavelet)
        logging.info('INFO: Obtained wavelet')
        return self.wavelet

    def correlate_wavelet_dataset(self):

        """Function correlates wavelet with complete set of filtered data
        :returns self.correlted_data: numpy 1D array containing correlation data
        """

        self.ma_dataset = np.ravel(self.ma_dataset)
        self.correlated_data = np.correlate(self.wavelet, self.ma_dataset)
        self.correlated_data = self.correlated_data - np.mean(self.correlated_data)
        self.correlated_data[self.correlated_data < 0] = 0
        return self.correlated_data

    def find_peaks(self, threshold_factor):

        """Function returns voltage at each beat
        :returns self.beat_voltage: array with voltages at beat
        """

        self.correlated_data = self.correlated_data
        self.threshold = (np.mean(self.correlated_data))*threshold_factor
        relative_maxima = argrelmax(self.correlated_data, order=30)
        relative_maxima = np.ravel(relative_maxima)
        self.beat_voltage = self.correlated_data[relative_maxima]
        self.beat_voltage = [i for i in self.beat_voltage if i >= self.threshold]
        return self.beat_voltage

    def find_peaks_index(self):

        """Function returns index for each beat voltage
        :returns self.peaks_index: numpy 1D array containing correlated and positive data
        """

        for i, j, in enumerate(self.beat_voltage):
            for p, q in enumerate(self.correlated_data):
                if q == self.beat_voltage[i]:
                    self.peaks_index.append(p)
        return self.peaks_index

    def beats_times(self):

        """Function returns a numpy array of the times at which a beat occurred
        :returns self.beats: times at which beats occur"""

        self.peaks_index = self.peaks_index
        self.beats = self.time_dataset[self.peaks_index]
        self.beats = np.ravel(self.beats)
        print('These are the times, in seconds, when the beats occurred:', self.beats)
        logging.info('INFO: Obtained beat times')
        return self.beats

    def num_beats(self):

        """Function returns the number of beats that occurred
        :returns len(self.beat_voltage): length of numpy array
        """

        self.beat_count = len(self.beat_voltage)
        print('This many beats occurred:', self.beat_count)
        logging.info('INFO: Obtained number of beats')
        return self.beat_count

    def duration(self):

        """Function returns time of ECG
        :returns max(self.time_dataset.values): time duration of ECG
        """

        self.dur = max(self.time_dataset)
        print('This is the time duration, in seconds, of the ECG:', self.dur)
        logging.info('INFO: Obtained ECG duration')
        return self.dur

    def voltage_extremes(self):

        """Function returns maximum and minimum voltage values in ECG
        :returns self.max_min: tuple containing maximum and minimum voltage values"""

        self.max_min = [self.voltage_dataset.max(), self.voltage_dataset.min()]
        print('The maximum and minimum voltage values are:', self.max_min)
        logging.info('INFO: Obtained voltage extrema')
        return self.max_min

    def mean_hr_bpm(self, user_input):

        """Function returns average heart rate over user specified input
        :param user_input: time in minutes
        :returns self.hr: average heart rate in beats per second"""

        if user_input*60 > max(self.time_dataset):
            self.hr = len(self.beat_voltage) / max(self.time_dataset)
        else:
            self.hr = len(self.beat_voltage)/(user_input*60)

        print('Average heart rate in beats/second is:', self.hr)
        logging.info('INFO: Obtained heart rate')
        return self.hr

    def make_json(self, file_name):

        """Function returns json file containing all required ECG Data
        :param file_name: name of file for data to be extracted from
        """

        file_contents = [{"Average Heart Rate": float(self.hr)},
                         {"Voltage Extrema": self.max_min},
                         {"ECG Duration": float(self.dur)},
                         {"Number of Beats": float(self.beat_count)},
                         {"Beat Times": np.asarray(self.beats).tolist()}
                         ]

        json_file_name = file_name.replace('.csv', '.json')
        json_file = open(json_file_name, 'w')
        json.dump(file_contents, json_file)
        json_file.write('\n')

        logging.info('INFO: Create .json file')
