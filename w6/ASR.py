import numpy as np
import pyaudio
from scipy.io import wavfile
import speech_recognition as sr

RATE = 16000
CHUNK = RATE*5

p=pyaudio.PyAudio()
print("Speak for 5 sec.")
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, output=True,frames_per_buffer=CHUNK, input_device_index=0)

data = np.frombuffer(stream.read(CHUNK),dtype=np.int16)
wavfile.write('D:/code/mmp/w6/hi~.wav',RATE, data.astype(np.int16))

r = sr.Recognizer()
wavr = sr.AudioFile('D:/code/mmp/w6/hi~.wav')
with wavr as source:
    audio = r.record(source)
    print(r.recognize_google(audio,language = 'ko'))

stream.stop_stream()
stream.close()
p.terminate()    
