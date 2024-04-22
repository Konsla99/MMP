#initialization

import sys
import pyaudio
import math
import struct
import time
#실제 시간을 원한다면
# import datetime

p = pyaudio.PyAudio()
def callback(in_data, frame_count, time_info, status):
    levels =[]
    for _i in range(1024):
        levels.append(struct.unpack('<h',in_data[_i:_i+2])[0])
    avg_chunk = sum(levels)/len(levels)
    #실제시간을 출력하기위함
    #now = datetime.datetime.now()
    #print(avg_chunk, now)
    print(avg_chunk, time_info['current_time'] )
    return (in_data, pyaudio.paContinue)    

#Open stream using callback
stream = p.open(format = pyaudio.paInt16,channels=1,rate=48000,frames_per_buffer=1024,\
                input=True,output=True,stream_callback=callback)

#Close the stream after 10 seconds
time.sleep(10)
stream.close()

