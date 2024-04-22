import numpy as np
import pyaudio
import keyboard
import sys

RATE = 48000
CHUNK = int(RATE/10)
HEAD = 10
DIS = 5
Vs = 339    #sonic
delaytype = 'a'

max_delay = int(RATE*2*HEAD*.01/Vs)
aheadL = np.zeros(max_delay, dtype=np.int16)
aheadR = np.zeros(max_delay, dtype=np.int16)
out = np.zeros(CHUNK*2, dtype=np.int16)

def Sound_rendering(signal, dir):
    distance, delay = Get_delay(dir)
    if dir >= 0:    #source가 오른쪽에 존재
        if delaytype == 'a':
            for i in range(CHUNK):
                if i < delay:
                    out[i*2] = aheadL[max_delay - delay + i]    #이전 buffer의 delayed sound
                else:
                    out[i*2] = signal[(i-delay)*2]  #ITD  
                    out[i*2] = int(out[i*2]*(DIS/(DIS+distance)))   #IID
                    out[i*2+1] = signal[i*2]    #오른쪽 소리는 원본 소리 그대로
        elif delaytype == 't':
            for i in range(CHUNK):
                if i < delay:
                    out[i*2] = aheadL[max_delay - delay + i]    #이전 buffer의 delayed sound
                else:
                    out[i*2] = signal[(i-delay)*2]  #ITD
                    out[i*2+1] = signal[i*2]    #오른쪽 소리는 원본 소리 그대로
        elif delaytype == 'i':
            for i in range(CHUNK):
                out[i*2] = signal[i*2]  #ITD
                out[i*2] = int(out[i*2]*(DIS/(DIS+distance)))   #IID
                out[i*2+1] = signal[i*2]    #오른쪽 소리는 원본 소리 그대로

                
    else:           #source가 왼쪽에 존재
        if delaytype == 'a':
            for i in range(CHUNK):
                if i < delay:
                    out[i*2+1] = aheadL[max_delay - delay + i]    #이전 buffer의 delayed sound
                else:
                    out[i*2+1] = signal[(i-delay)*2]  #ITD  
                    out[i*2+1] = int(out[i*2+1]*(DIS/(DIS+distance)))   #IID
                    out[i*2] = signal[i*2]    #왼쪽 소리는 원본 소리 그대로
        elif delaytype == 't':
            for i in range(CHUNK):
                if i < delay:
                    out[i*2+1] = aheadL[max_delay - delay + i]    #이전 buffer의 delayed sound
                else:
                    out[i*2+1] = signal[(i-delay)*2]  #ITD
                    out[i*2] = signal[i*2]    #왼쪽 소리는 원본 소리 그대로
        elif delaytype == 'i':
            for i in range(CHUNK):
                out[i*2+1] = signal[i*2]  #ITD
                out[i*2+1] = int(out[i*2+1]*(DIS/(DIS+distance)))   #IID
                out[i*2] = signal[i*2]    #왼쪽 소리는 원본 소리 그대로


    for i in range(max_delay):  #buffer를 넘어간 delayed sound
        aheadL[i] = signal[(CHUNK - max_delay + i)*2]
        aheadR[i] = signal[(CHUNK - max_delay + i)*2]
    return out

def Get_delay(dir_angle):   #머리 크기와 각도로 거리 차이와 시간 차이 계산함수
    distance = 2 * HEAD * 0.01 * np.abs(np.sin(dir_angle))
    delay = int(distance * RATE/Vs)
    return distance, delay

p=pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=2, rate=RATE, input=True, output=True,frames_per_buffer=CHUNK, input_device_index=1)

rendering = True
target_dir = np.pi*float(sys.argv[1])/180
dir = target_dir

while(True):
    samples = stream.read(CHUNK)
    in_data = np.fromstring(samples, dtype=np.int16)
    out = Sound_rendering(in_data, dir)
    y = out.tostring()
    stream.write(y)
    if keyboard.is_pressed('q'):
        break
    if keyboard.is_pressed('s'):
        if rendering:
            rendering = False
            dir = 0
            print("Sound rendering off.")
        else:
            rendering = True
            dir = target_dir
            print("Sound rendering on.")
    if keyboard.is_pressed('a'):    #all
        delaytype = 'a'
        print("IID + ITD")
    elif keyboard.is_pressed('i'or'I'):  #only intensuty difference
        delaytype = 'i'
        print("IID")
    elif keyboard.is_pressed('t'or'T'):  #only time difference
        delaytype = 't'
        print("ITD")

stream.stop_stream()
stream.close()
p.terminate()
