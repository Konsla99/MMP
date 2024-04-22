import speech_recognition as sr
from gtts import gTTS
import soundfile as sf
from scipy import signal
import playsound
from scipy.io import wavfile
import numpy as np
import pyaudio
import keyboard
import sys
import os
import time

# 키보드 인터럽트를 통해 언어 설정 가능
RATE = 16000
CHUNK = RATE * 5
file_index = 0  # 파일 인덱스 초기화

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=0)

r = sr.Recognizer()
lang = "ko-KR"  # 초기 언어 설정

while True:
    print("Speak for 5 sec.")
    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    temp_filename = f'D:/code/mmp/w6/temp{file_index}.wav'
    response_filename = f'D:/code/mmp/w6/response{file_index}.wav'

    wavfile.write(temp_filename, RATE, data)
    wavr = sr.AudioFile(temp_filename)

    with wavr as source:
        audio = r.record(source)
        try:
            recognized_text = r.recognize_google(audio, language=lang)
            print("You said: " + recognized_text)

            if recognized_text == '안녕':
                response_text = "안녕하세요" if lang == 'ko-KR' else "Hi"
            elif recognized_text == '멀티미디어 프로그램':
                response_text = "재미있는 수업" if lang == 'ko-KR' else "Fun class"
            elif recognized_text == '비가 온다':
                response_text = "우산을 챙기세요"
            else:
                response_text = "잘 이해하지 못했습니다"
        except sr.UnknownValueError:
            response_text = "인식할 수 없는 음성입니다"
        except sr.RequestError as e:
            response_text = f"요청 오류가 발생했습니다; {e}"

        print("Response: " + response_text)
        tts = gTTS(text=response_text, lang=lang[:2])
        tts.save(response_filename)
        d, fs = sf.read(response_filename)
        ds = signal.resample(d, int(len(d) * RATE / fs))
        sf.write(response_filename, ds, RATE)
        playsound.playsound(response_filename)

        file_index += 1  # 파일 인덱스 증가

        if keyboard.is_pressed('q'):
            print("Exiting program.")
            break

        if keyboard.is_pressed('l'):
            lang = 'en' if lang == 'ko-KR' else 'ko-KR'
            print(f"Language set to: {lang}")

stream.stop_stream()
stream.close()
p.terminate()
