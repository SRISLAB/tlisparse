from utils.pybearing import pybearing 
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats

if __name__ == '__main__':
    model = pybearing()
    time_tot = 1         # signal time duration 
    fs = 20000           # signal sampling rate                   
    oc = 3               # order characteristic frequency (fault characteristic frequency/bearing frequency)
    fn = 3000            # natural frequency of the system 
    decay_f = 1          # decay factor        
    n_samples = 5       # total number of samples to generate (for both healthy and faulty classes)
    speed_range = np.array([5,10])        # the range of bearing speed. the bearing speeds will e generated randomly in this range    
    data = model.signal_gen(time_tot, fs, oc, fn, decay_f, n_samples, speed_range)
    saveing_size =1000         # the size of each data after processing
    samples = 5000            # the length of each sub signal in the data augmentation process
    stride = 1000             # the stride size between each to sequential sub signals. by decreasing this size, the number of sub signals will increase. 
    upfactor = 5               # upfactor for the computed order tracking process 
    #processed_data = model.signal_analyser(data, saveing_size, samples, stride, upfactor)

    signal = data[0]
    kuters = scipy.stats.kurtosis(abs(signal))

    plt.plot(kuters)
    plt.show()


 

    print(len(data[0]))
 