import speech_recognition as sr
from gtts import gTTS
import soundfile as sf
from scipy import signal
import playsound

s=input("Enter string:")
tts =gTTS(text = s, lang ='ko')
tts.save('D:/code/mmp/w6/voice.wav')

d, fs= sf.read('D:/code/mmp/w6/voice.wav')
ds = signal.resample(d,int(len(d)*16/24))
sf.write('D:/code/mmp/w6/voice.wav',ds, 16000)

playsound.playsound('D:/code/mmp/w6/voice.wav')