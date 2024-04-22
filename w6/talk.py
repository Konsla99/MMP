from pynput import keyboard
import speech_recognition as sr
from gtts import gTTS
import soundfile as sf
from scipy import signal
import playsound
from scipy.io import wavfile
import numpy as np
import pyaudio
import time

# 초기 설정
RATE = 16000
CHUNK = RATE * 5
file_index = 0
lang = "ko-KR"
run = True

def on_press(key):
    global lang, run
    try:
        if key.char == 'q':
            print("Exiting program.")
            return False  # 리스너 종료
        elif key.char == 'l':
            lang = 'en' if lang == 'ko-KR' else 'ko-KR'
            print(f"Language set to: {lang}")
        elif key.char == 'r':
            run = not run  # 실행 상태 토글
    except AttributeError:
        pass

def main():
    global file_index
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=0)
    r = sr.Recognizer()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while listener.running:
            if run:
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
                            response_text = "안녕하세요" 
                        elif recognized_text == '밥은 먹었니':
                            response_text = "네 그쪽은요" 
                        elif recognized_text == '나도 먹었어':
                            response_text = "다행이네요"
                        elif recognized_text == 'annyeong' and lang =='en':
                            response_text = "hi"   
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

                    file_index += 1
            time.sleep(0.1)  # CPU 사용량을 관리
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        listener.stop()

if __name__ == '__main__':
    main()
