import numpy as np
import pyaudio
import keyboard
import random

RATE = 48000
CHUNK = int(RATE / 10)  # Chunk size
HEAD = 10  # Head size in cm
DIS = 1  # Distance
Vs = 339  # Speed of sound in m/s

max_delay = int(RATE * 2 * HEAD * 0.01 / Vs)
aheadL = np.zeros(max_delay, dtype=np.int16)
aheadR = np.zeros(max_delay, dtype=np.int16)
out = np.zeros(CHUNK * 2, dtype=np.int16)

def Sound_rendering(signal, dir):
    distance, delay = Get_delay(dir)
    if dir >= 0:    # Source is on the right
        for i in range(CHUNK):
            if i < delay:
                out[i * 2] = aheadL[max_delay - delay + i]  # Previous buffer's delayed sound
            else:
                out[i * 2] = signal[(i - delay) * 2]  # ITD
                out[i * 2] = int(out[i * 2] * (DIS / (DIS + distance)))  # IID
                out[i * 2 + 1] = signal[i * 2]  # Right sound remains original
    else:           # Source is on the left
        for i in range(CHUNK):
            if i < delay:
                out[i * 2 + 1] = aheadR[max_delay - delay + i]  # Previous buffer's delayed sound
            else:
                out[i * 2 + 1] = signal[(i - delay) * 2]  # ITD
                out[i * 2 + 1] = int(out[i * 2 + 1] * (DIS / (DIS + distance)))  # IID
                out[i * 2] = signal[i * 2]  # Left sound remains original
    for i in range(max_delay):  # Update delayed sound for the next buffer
        aheadL[i] = signal[(CHUNK - max_delay + i) * 2]
        aheadR[i] = signal[(CHUNK - max_delay + i) * 2 + 1]
    return out

def Get_delay(dir_angle):
    distance = 2 * HEAD * 0.01 * np.abs(np.sin(dir_angle))  # Calculate distance difference based on head size and angle
    delay = int(distance * RATE / Vs)  # Calculate time difference
    return distance, delay

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=2, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)

correct_count = 0  # Count of correct guesses
angle_errors = []  # Store angle errors

for _ in range(10):  # Play the game 10 times
    rand_dir = random.randint(-90, 90)  # Random direction in degrees
    target_dir = np.pi * float(rand_dir) / 180  # Convert to radians
    correct_answer = 'l' if rand_dir < 0 else 'r'  # Determine the correct answer based on the direction

    print("\nPress 'q' when ready to guess the direction...")
    while True:
        # Stream audio processing here (omitted for brevity)
        if keyboard.is_pressed('q'):  # Break the loop to guess after pressing 'q'
            break

    direction_guess = input("Guess the Direction (L/R): ").lower()
    angle_guess = int(input("Guess the Angle (-90 to 90): "))
    angle_error = abs(rand_dir - angle_guess)  # Calculate the absolute error between the guessed angle and the actual angle
    
    if direction_guess == correct_answer:
        correct_count += 1
        print("Correct direction!")
    else:
        print("Wrong direction!")
    print(f"Actual angle: {rand_dir}°, Your guess: {angle_guess}°, Error: {angle_error}°")
    angle_errors.append(angle_error)  # Store the error

print(f"\nGame Over. Correct Answers: {correct_count}/10")
if angle_errors:
    print("Angle Errors: ", angle_errors)

stream.stop_stream()
stream.close()
p.terminate()
