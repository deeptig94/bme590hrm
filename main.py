from heart_rate_monitor import ECGData


x = ECGData('test_data/test_data1.csv')

ma = x.moving_average()
fm = x.find_max_voltage_index()
gw = x.get_wavelet()
cw = x.correlate_wavelet_dataset()
fp = x.find_peaks(7)
fpi = x.find_peaks_index()
bt = x.beats_times()
nb = x.num_beats()
dur = x.duration()
ve = x.voltage_extremes()
hr = x.mean_hr_bpm(user_input=2)
x.make_json('test_data/test_data1.csv')

