import wave

with wave.open('test.wav', 'r') as wav_file:
    nchannels = wav_file.getnchannels()
    print(f'The file has {nchannels} channels.')  # '1'이면 모노, '2'면 스테레오