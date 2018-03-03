from heart_rate_monitor import ECGData

x = ECGData('test_data/test_data1.csv')
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


