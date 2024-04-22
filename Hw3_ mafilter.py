# movig average filter  + window
import numpy as np
import pyaudio
import time
import sys
import keyboard
import scipy.signal as signal
import fplot

#sampling freq
RATE = 16000
#size of buff 0.1sec  data
CHUNK = int(RATE/10)
# kerbel_size는 tap 수를 의미하므로 f
# 이 부분을 조정하여 tap 수를 조정하면 된다 
# Tap 수는 홀수여야 할것
kernel_size = 505

kernel = np.full(kernel_size,1/kernel_size)
in_data = np.zeros(CHUNK+kernel_size, dtype=np.int16)
# in_data = np.zeros(CHUNK+kernel_size-1, dtype=np.int16)

filter_on = False

# def filter
# cut off freq = nomalized 
# sampling frequency is 16khz  0.125= 1khz
rect_kernel = signal.firwin(kernel_size,cutoff=0.125,window = "boxcar")
hamm_kernel = signal.firwin(kernel_size,cutoff=0.125,window = "hamming")


# plot the filter
print("Plot of Rectangular Filter")
fplot.mfreqz(rect_kernel)
fplot.show()

print("Plot of Hamming Filter")
fplot.mfreqz(hamm_kernel)
fplot.show()

# other filters ramdom window 

# window select hanning,Black man
#highpass  f_cutoff is 1khz
highpass = signal.firwin(kernel_size,cutoff=0.125,window="hann",pass_zero='highpass')
# BPF & BSF f_cutoff = 500hz ~ 1.5khz
#bandpass
bandpass = signal.firwin(kernel_size, [0.0625, 0.1875], window="hann", pass_zero='bandpass')
#bandstop
bandstop = signal.firwin(kernel_size, [0.0625, 0.1875], window="hann", pass_zero='bandstop')

kernel = bandstop

# real time 처리를 고려

def convolution(signal, kernel):
    n_sig = signal.size
    n_ker = kernel.size
    n_conv = n_sig - n_ker

    #by favtor of 3
    rev_kernel =kernel[::-1].copy()
    result = np.zeros(n_conv,dtype=np.int16)

    for i  in range(n_conv):
        if filter_on:
            result[i] = np.dot(signal[i:i+ n_ker],rev_kernel)
        else:
            result[i] =  signal[i+n_ker]
    
    signal[0:n_ker] = signal[n_sig-n_ker:n_sig] 

    return result


# main routine

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,output=True,
                frames_per_buffer=CHUNK,input_device_index=2)


while(True):
    # CHUNK SIZE만큼 버퍼에 in
    # samples = input buff  string type으로 들어감 binary data로 변경필요
    samples = stream.read(CHUNK)
    # fromstring을 통해 받아와 변화 int16으로 
    in_data[kernel_size:kernel_size+CHUNK] = np.fromstring(samples, dtype=np.int16)
    out = convolution(in_data,kernel)
    # out을 파일로 쓰면 파형 관측 가능
    # tostring을 통해 내보내야함  output도 string으로 나가야함
    y = out.tostring()
    stream.write(y)

    if keyboard.is_pressed('q'):
        break
    if keyboard.is_pressed('f'):
        if filter_on:
            filter_on = False
            print("Filter off.")
        else:
            filter_on = True
            print("Filter on.")    
# terminate program
stream.stop_stream()
stream.close()

p.terminate
