import numpy as np
import pyaudio
import keyboard
import random

RATE = 48000
CHUNK = int(RATE/10)
HEAD = 10
DIS = 1
Vs = 339    #sonic
ans = 0

max_delay = int(RATE*2*HEAD*.01/Vs)
aheadL = np.zeros(max_delay, dtype=np.int16)
aheadR = np.zeros(max_delay, dtype=np.int16)
out = np.zeros(CHUNK*2, dtype=np.int16)

def Sound_rendering(signal, dir):
    distance, delay = Get_delay(dir)
    if dir >= 0:    #source가 오른쪽에 존재
        for i in range(CHUNK):
            if i < delay:
                out[i*2] = aheadL[max_delay - delay + i]    #이전 buffer의 delayed sound
            else:
                out[i*2] = signal[(i-delay)*2]  #ITD  
                out[i*2] = int(out[i*2]*(DIS/(DIS+distance)))   #IID
                out[i*2+1] = signal[i*2]    #오른쪽 소리는 원본 소리 그대로
                
    else:           #source가 왼쪽에 존재
        for i in range(CHUNK):
            if i < delay:
                out[i*2+1] = aheadL[max_delay - delay + i]    #이전 buffer의 delayed sound
            else:
                out[i*2+1] = signal[(i-delay)*2]  #ITD  
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
stream = p.open(format=pyaudio.paInt16, channels=2, rate=RATE, input=True, output=True,frames_per_buffer=CHUNK, input_device_index=0)

correct_count = 0  # 맞힌 횟수
wrong_angles = []  # 틀린 각도 저장
angle_errors = []  # Store angle errors

for _ in range(10):  # 게임을 10번 반복

    rendering = True
    rand_dir = random.randint(-90,90)
    target_dir = np.pi*float(rand_dir)/180
    dir = target_dir

    # Determine correct answer based on the direction
    correct_answer = 'l' if rand_dir < 0 else 'r'

    print("press 'q' to guess the answer")
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

    guess = input("Guess the Direction (L/R): ").lower()
    angle_guess = int(input("Guess the Angle (-90 to 90): "))
    angle_error = abs(rand_dir - angle_guess)  # Calculate the absolute error between the guessed angle and the actual angle
    max_error = 180# 오차 백분율 계산
    error_percentage = (angle_error / max_error) * 100

    if guess == correct_answer:
        correct_count += 1
        print("Correct!!!")
        print(f"{rand_dir}")

    else:
        wrong_angles.append(rand_dir)
        print(f"Wrong, Answer is {correct_answer}")
        print(f"{rand_dir}")

    # 이 부분은 방향 맞추기에 상관없이 항상 실행됩니다.
    print(f"실제 각도: {rand_dir}도")
    angle_errors.append(angle_error)  # 오차를 리스트에 추가
    print(f"입력한 각도와 실제 각도 간의 오차: {angle_error}도 (오차율: {error_percentage:.2f}%)")
print(f"Correct Answers: {correct_count}")
print("Wrong Angles: ", wrong_angles)

stream.stop_stream()
stream.close()
p.terminate()
