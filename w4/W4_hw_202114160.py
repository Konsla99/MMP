import numpy as np
from scipy import signal
from scipy.signal import get_window
from scipy.io import wavfile
import sys

WL = 256
WR = 128

# x= real array wlen= window length, step = window shift step, w= window 
def stft(x, w, step):
    wlen = len(w)
    nsample = len(x)
    # 1e-12~~ = 0값을 피하기위한 극소값
    Xtf = np.array([np.fft.rfft(w*x[i:i+wlen]) 
                    for i in range(0,nsample - wlen+1, step)])+ 1e-12*(1+1j)
    return Xtf 
# inverse short time F-T
def istft(Xtf, w,step, nsample):
    nframe = len(Xtf)
    wlen =len(w)

    y = np.zeros(nsample)
    ws = np.zeros(nsample)

    ind = 0
    for i in range(0, nsample-wlen +1, step):
        y[i:i+wlen] += w*np.fft.irfft(Xtf[ind])
        ws[i:i+wlen] += w*w
        ind +=1
    ws[ws==0] = 1
    y = y/ws

    return y

def filter_ft(mag,fcut,fcut2,ftype):
    #ft_bin = 
    Nf,ft_bin = mag.shape
    fmag = np.zeros([Nf,ft_bin])
    # 필터의 상승 하강 엣지 정의
    fcut_pos = int(ft_bin*fcut)
    fcut_neg = int(ft_bin*fcut2)
    # filter type
    if ftype == 'lowpass':
        fmag[:,0:fcut_pos]=mag[:,0:fcut_pos]
    elif ftype == 'highpass':    
        fmag[:,fcut_pos:ft_bin]=mag[:,fcut_pos:ft_bin]
    elif ftype == 'bandpass':
        fmag[:, fcut_pos:fcut_neg]= mag[:, fcut_pos:fcut_neg]
    elif ftype == 'bandstop':
        fmag[:,0:fcut_pos]= mag[:,0:fcut_pos]
        fmag[:,fcut_neg:ft_bin]= mag[:,fcut_neg:ft_bin]
    else:
        print("error can't find filter")
        return -1    
    return fmag    

# 명령어 인자 순서  1. input 2. output  3.filter 4.fcut 5.fcut2

fs, data = wavfile.read(sys.argv[1])
fcut = float(sys.argv[4])
w = get_window('hann', WL)
Xf =stft(data,w, WR)
Mag = np.abs(Xf)
Phs = np.angle(Xf)

# 인자가 여러개인 경우 = cutoff 주파수 값이 2개 이상인경우
if len(sys.argv)>5:
    fcut2 = float(sys.argv[5])
else:
    fcut2 = 1    
fMag = filter_ft(Mag, fcut,fcut2,sys.argv[3])
Xfr = fMag*np.exp(1j * Phs)

y = istft(Xfr,w,WR,len(data))
wavfile.write(sys.argv[2],fs,y.astype(np.int16))