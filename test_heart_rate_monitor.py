import json

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


def test_num_beats():
    f = a.num_beats()
    assert(f == 34)


def test_duration():
    g = a.duration()
    assert(g == 27.775)


def test_voltage_extremes():
    h = a.voltage_extremes()
    assert(h == [1.05, -0.68])


def test_mean_hr_bpm():
    p = a.mean_hr_bpm(2)
    assert(abs(p - 1.22412241) < 0.001)


def test_make_json():
    a.make_json('test_data/test_data1.csv')
    z = json.loads(open('test_data/test_data1.json').read())
    assert(z[0] == {"Average Heart Rate": 1.2241224122412242})
