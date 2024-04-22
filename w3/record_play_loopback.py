# This is FIR filter for MMp W3 HW
# made by mingi chae 202114160
# import sys
import numpy as np
import pyaudio
import keyboard

#loopback code recording & playing

#sampling freq
RATE = 16000
#size of buff 0.1sec  data
CHUNK = int(RATE/10)
#sys.path.append('c:/users/vtoree/anaconda3/lib/site-packages')

#process is  important point
# signal is np.array
def process(signal):
    result = signal# 입력을 출력으로 그대로 전송
    return result

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,output=True,
                frames_per_buffer=CHUNK,input_device_index=0)

while(True):
    # CHUNK SIZE만큼 버퍼에 in
    # samples = input buff  string type으로 들어감 binary data로 변경필요
    samples = stream.read(CHUNK)
    # fromstring을 통해 받아와 변화 int16으로 
    in_data = np.fromstring(samples, dtype=np.int16)
    out = process(in_data)
    # out을 파일로 쓰면 파형 관측 가능
    # tostring을 통해 내보내야함  output도 string으로 나가야함
    y = out.tostring()
    stream.write(y)
    if keyboard.is_pressed('q'):
        break
stream.stop_stream()
stream.close()
p.terminate
