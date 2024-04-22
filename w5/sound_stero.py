import numpy as np
import pyaudio
import keyboard

RATE = 48000
CHUNK = int(RATE/1000)

output = np.zeros(CHUNK*2, dtype=np.int16)

def Select_channel(signal, ch): #출력을 좌,우 중 어디로 할지 결정, 한쪽만 나옴
    for i in range(CHUNK):
        if ch=='left':
            output[i*2]=signal[i*2]
            output[i*2+1]=0
        else:
            output[i*2]=0
            output[i*2+1]=signal[i*2]
    return output


p=pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=2, rate=RATE, input=True, output=True,frames_per_buffer=CHUNK, input_device_index=0)
state = 0
while(True):
    samples = stream.read(CHUNK)
    in_data = np.fromstring(samples, dtype=np.int16)
    
    if state == 0:
        out = Select_channel(in_data, 'left')
    elif state == 1:
        out = Select_channel(in_data, 'right')

    if keyboard.is_pressed('l'):
        state = 1
        print("right")
    elif keyboard.is_pressed('r'):
        state = 0
        print("left")
        
    y = out.tostring()
    stream.write(y)
    if keyboard.is_pressed('q'):
        break
 
stream.stop_stream()
stream.close()
p.terminate()
