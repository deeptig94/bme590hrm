import pytest
import pandas as pd
import numpy as np
from heart_rate_monitor import ECGData

a = ECGData('test_data/test_data1.csv')


def test_moving_average():
    b = a.moving_average()
    assert(abs(b[0]-[-0.04833333]) > -0.001)


def test_find_max_voltage_index():
    c = a.find_max_voltage_index()
    assert(c == [7392])


def test_get_wavelet():
    d = a.get_wavelet()
    assert(abs(d[0]-[-0.11166667]) > -0.001)


def test_beats_times():
    a.correlate_wavelet_dataset()
    a.find_peaks(7)
    a.find_peaks_index()
    e = a.beats_times()

    assert(e[0] == 0.722)







