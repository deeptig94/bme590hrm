import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import autocorrelation_plot


class ECGData:

    def __init__(self, file_name):
        self.dataset = 0
        self.max_voltage = 0
        self.file_name = file_name
        #self.voltage = voltage
        #self.time = time

    def read_data(self, file_name):
        self.dataset = pd.read_csv(file_name)
        self.dataset.columns = ['time', 'voltage']
        return self.dataset

    def plot_data(self):
        self.dataset.columns = ['time', 'voltage']
        plt.title("Heart Rate Signal")
        self.dataset.plot(x='time', y='voltage')
        plt.show()


    def find_max_voltage(self):
        self.max_voltage = self.dataset['voltage'].idxmax()
        print(self.max_voltage)
        return self.max_voltage

    '''def auto_correlate(self):
        #print(dataset['voltage'].autocorr(lag=1))
        autocorrelation_plot(self.dataset)
        plt.show()'''


x = ECGData('test_data/test_data1.csv')
x.read_data('test_data/test_data1.csv')
x.plot_data()
x.find_max_voltage()
#x.auto_correlate()