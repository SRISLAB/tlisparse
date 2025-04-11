#########################################################################      
##   this code performs a fault detection strategy using deep learning         
##   on the rolling element bearing vibration data.                            
##   Form more information on our apraoch and the algorithm please see         
##   our published article:                                                    
##   Sadoughi, Mohammakazem, Austin Downey, Garrett Bunge, Aditya Ranawat,     
##   Chao Hu, and Simon Laflamme. "A Deep Learning-based Approach for Fault    
##   Diagnosis of Roller Element Bearings." (2018).                            
########################################################################


from scipy.signal import hilbert
import numpy as np
from scipy import interpolate
import scipy.signal as sig
from scipy.fftpack import fft
from scipy import integrate

from sklearn.preprocessing import OneHotEncoder
import random



## the computed order tracking prgram. this code transfers the signal from the time domain to the phase domain. 
def COT(data, speed, upfactor, fs):
    
    samples = np.shape(data)[0]     # extract the number of samples   
    t = np.arange(0.0, samples/fs, 1/fs)    # generate a time step data
    
    # resample the data to the higher sampling rate. This can increase the accuracy of interpolation which 
    # will be used in the next steps. 
    dataUp = sig.resample(data, samples * upfactor)      
    timeUp = np.arange(0.0, samples/fs, 1/(fs*upfactor))
    
    f = interpolate.interp1d(t, speed, fill_value="extrapolate")       # interpolation function for the speed data
    speedUp = f(timeUp)                                                # result of interpolation
    
    phase_int = integrate.cumtrapz(speedUp, timeUp, initial=0)         # build another interpolation function for the phase
    phase = np.arange(0.0, phase_int[-1], phase_int[-1]/(samples*upfactor))  # result  of interpolation function
    
    f = interpolate.interp1d(phase_int, dataUp) 
    data_ordered = f(phase) 
    
    return data_ordered, phase                                         # store the data and its phase 



## This function has been written for generating the random batch of samples from the data for stochastic optimization. 
            # it will be used in the training process of deep learning model
def next_batch(x, y, batch_size):    # x is the input, y is the output and batch_size is the size of batch (number of samples)
    num_elements = x.shape[0]
    new_index = np.arange(num_elements)                                        # generate random index
    np.random.shuffle(new_index)                                               # shuffle data
    new_index = new_index [:batch_size]                                        # select the first batch_size number of samples
    
    
    # the output will be the batch of input and output which have been chosen randomly
    x = x[new_index,:]
    y = y[new_index]
    return x, y


        
class pybearing:
    
    def __init__(self):
        return None
    
    ## loading the data with the numpy format
    def load_data(self, fs, dir_data):
        self.fs = fs         
        data = np.load(dir_data)
        self.n_samples = len(data)
        return data

    ## generating the simulated vibration data from the beating with both helthy and faulty condition
    def signal_gen(self, time_tot, fs = 20000, oc = 3, fn = 3000, decay_f = 2, n_samples = 200, speed_range = np.array([10,20])):
        
        
        self.time_tot = time_tot
        self.fs = fs
        self.oc = oc
        self.fn = fn
        self.decay_f = decay_f
        self.n_samples = n_samples
        self.speed_range = speed_range
        
        # generate the time steps
        t = np.linspace(0, self.time_tot, num = self.fs*self.time_tot)
        
        # initialize the data: data.shape = number of samples * Len(t) + 2. the last column of data represents the data label 
        # (e.g. label = 1 for faulty signal and label = 0 for healthy signal). the one before the last column shows the speed data
        data = np.zeros([self.n_samples,len(t)+2])
        
        for k in range(self.n_samples):
            
            # Randomly chose the fault conidtion. label = 1: faulty signal and label = 0: healthy signal
            label = random.randint(0, 1) 
            fr = random.randint(self.speed_range[0], self.speed_range[1])   # randomly select the speed
            
            if label == 1:     # faulty signal
                signal = np.zeros(len(t))    # initialize the signal value
                t_fault = np.linspace(0, self.time_tot, num = self.oc*fr*self.time_tot)
                
                for i in t_fault:
                    signal += np.heaviside(t-i, 1.0)*np.exp(-1*self.decay_f*self.fs*(t-i)**2)   
                    
                signal = signal * np.sin(2*np.pi*self.fn*t) * np.sin(2*np.pi*fr*t)
                signal += 0.2*np.random.normal(0,1,len(t))
                data[k,0:len(t)] = signal
                data[k,len(t)] = fr
                data[k,len(t)+1] = label
            else:   
                signal = np.sin(2*np.pi*self.fn*t) * np.sin(2*np.pi*fr*t)
                signal += 0.2*np.random.normal(0,1,len(t))
                data[k,0:len(t)] = signal
                data[k,len(t)] = fr
                data[k,len(t)+1] = label
                
                
            if 10 * k % self.n_samples == 0: 
                print ("Progress: %03d out of %03d samples have been generated" % (k, self.n_samples))
        print ("All samples have been generated succesfully") 
        return data
    
    ## performing several signal processing techniques to denoise and filter the signal and transfer it to the order domain.
    def signal_analyser(self, data, saveing_size =1000, samples = 2000, stride = 3000, upfactor = 5):
        
        self.saveing_size = saveing_size
        self.samples = samples
        self.stride = stride
        self.upfactor = upfactor  
        
        reference_order = np.linspace(0.0, 10, self.saveing_size)
        
        processed_data = np.zeros([0, self.saveing_size+1])
        signal_len = data.shape[1]-2
        
        for i in range(self.n_samples):        
            # adding the spead data
            speed = data[i,-2]*np.ones(signal_len)
            signal = data[i,0:-2]
            # slicing the signal
            first_index = 0
            last_index = first_index + self.samples
            
            while last_index < signal_len: 
            
                subsignal = signal[first_index:last_index]
                subspeed = speed[first_index:last_index]        
                first_index = first_index + self.stride
                last_index = first_index + self.samples
                # COT analysis 
                signal_ordered = np.empty([subsignal.shape[0],2])
                signal_ordered[:,1], signal_ordered[:,0] = COT(subsignal, subspeed, 1, self.fs)   
                
                # envelope analsis 
                amplitude_envelope = np.empty(signal_ordered.shape)
                amplitude_envelope[:,1] = np.abs(hilbert(signal_ordered[:,1]))
                amplitude_envelope[:,0] = signal_ordered[:,0]   
                         
                #FFT analysis
                orer_s = len(amplitude_envelope[:,0])/(amplitude_envelope[-1,0]-amplitude_envelope[0,0])
                ps_signal_ordered = np.empty([len(amplitude_envelope[:,0])//2, amplitude_envelope.shape[1]])
                ps_signal_ordered [:,0] = np.linspace(0.0, orer_s/2, len(amplitude_envelope[:,0])//2)
                ps_signal = fft(amplitude_envelope[:,1])
                ps_signal_ordered [:,1] = np.abs(ps_signal[0:self.samples//2])  
                
                ps_signal_final = np.empty([self.saveing_size, amplitude_envelope.shape[1]])
                ps_signal_final [:,0] = reference_order
                f = interpolate.interp1d(ps_signal_ordered[:,0], ps_signal_ordered[:,1])
                ps_signal_final[:,1] = f(ps_signal_final [:,0]) 
                new_data = np.concatenate((np.expand_dims( ps_signal_final [:,1], axis=0) , np.expand_dims(np.expand_dims(data[i,-1], axis=0), axis=0)), axis=1)    
                processed_data  = np.concatenate((processed_data, new_data), axis=0) 
            if 10 * i % self.n_samples == 0: 
                print ("Progress: %03d out of %03d samples have been processed" % (i, self.n_samples))   
        print ("All samples have been processed succesfully") 
        return processed_data
                 
   



            
