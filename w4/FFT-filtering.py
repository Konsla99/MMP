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
    
    Xtf = np.array([np.fft.rfft(w*x[i:i+wlen]) for i in range(0,nsample - wlen+1, step)])+ 1e-12*(1+1j)
    return Xtf 

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

def filter_ft(mag,fcut,ftype):
    Nf,ft_bin = mag.shape
    fmag = np.zeros([Nf,ft_bin])
    fcut_pos = int(ft_bin*fcut)

    if ftype == 'lowpass':
        fmag[:,0:fcut_pos]=mag[:,0:fcut_pos]
    if ftype == 'highpass':    
        fmag[:,fcut_pos:ft_bin]=mag[:,fcut_pos:ft_bin]

    return fmag    

fs, data = wavfile.read(sys.argv[1])
fcut = float(sys.argv[3])
w = get_window('hann', WL)
Xf =stft(data,w, WR)
Mag = np.abs(Xf)
Phs = np.angle(Xf)
fMag = filter_ft(Mag, fcut,sys.argv[4])
Xfr = fMag*np.exp(1j * Phs)
y = istft(Xfr,w,WR,len(data))
wavfile.write(sys.argv[2],fs,y.astype(np.int16))