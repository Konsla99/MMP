#low level implementation 1d-convolution
# import numpy as np

# def convolve_1d(signal, kernel):
#     n_sig = signal.size
#     n_ker = kernel.size
#     n_conv = n_sig - n_ker + 1

#     #by favtor of 3
#     rev_kernel =kernel[::-1].copy
#     result = np.zeros(n_conv,dtype=np.double)
#     for i  in range(n_conv):
#         result[i] = np.dot(signal[i: i+ n_ker],rev_kernel)
#     return result


# movig average filter  +
import numpy as np
import pyaudio
import time
import sys
import keyboard

#sampling freq
RATE = 16000
#size of buff 0.1sec  data
CHUNK = int(RATE/10)
kernel_size = 9
kernel = np.full(kernel_size,1/kernel_size)
in_data = np.zeros(CHUNK+kernel_size, dtype=np.int16)
# in_data = np.zeros(CHUNK+kernel_size-1, dtype=np.int16)

filter_on = False

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
                frames_per_buffer=CHUNK,input_device_index=0)


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
