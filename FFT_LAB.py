import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import scipy.signal as signal

#Normalization
def Normalization(audio):
    '''
    Normalizate the audio
    :return:
    normal_audio
    '''
    max_ampltitude= max(abs(audio))
    normal_ampltitude = audio/max_ampltitude
    return normal_ampltitude

#FFT
def FFT_Audio(audio,sample_rate):
    '''
    Audio FFT
    :return: fft_audio, fft_frequs
    '''
    fft_audio = nf.fft(audio)
    fft_freqs = nf.fftfreq(len(audio),1/sample_rate)
    return fft_audio,fft_freqs

#Improve Quality
def Improve_Quality():
    return

#educe amplitudes
def audio_compression(audio_data, threshold=0.2):
    # compute amplitude
    amplitude = np.abs(audio_data)
    # educe amplitudes greater than the threshold
    compressed_audio = np.where(amplitude > threshold, threshold, amplitude)
    # Preserve the sign of the original signal
    compressed_audio = compressed_audio * np.sign(audio_data)
    return compressed_audio


#Remove Noise by myself
def Remove_Noise(fft_audio,threshold):
    fft_audio[np.abs(fft_audio)>threshold] = 0
    fft_audio_new = np.abs(fft_audio)
    return fft_audio_new

#Save Voice
def Save_Voice(sample_rate,filter_sigs,path):
    wf.write(path, sample_rate, filter_sigs)
    return None

if __name__ == '__main__':
    sample_rate, audio = wf.read("/Users/haoqinghua/Desktop/1cm_voice.wav")  # add address of wav profile
    print(sample_rate)
    print(audio.shape)
    times = np.arange(audio.size) / sample_rate
    # Normalization
    normal_amplitude=  Normalization(audio)
    #Plot1_Matplotlib_Time_domin_picture
    time_domin_picture = plt.figure(figsize=(16,10),dpi=600)
    plt.title("Time Domain", fontsize=26)
    plt.xticks(fontsize=14,color='black')
    plt.yticks(fontsize=14,color="black")
    plt.xlabel("Time(s)",fontsize=18)
    plt.ylabel("Signal",fontsize=18)
    plt.grid(ls='--', lw=1, c='gray')
    # plt.plot(times[int(times.size/4):3*int(times.size/4)],normalized_audio[int(times.size/4):3*int(times.size/4)],c="orangered",label="Noised")
    plt.plot(times, normal_amplitude)
    # time_domin_picture.savefig(fname="/Users/haoqinghua/Desktop/python/DSP/Time_Domin.png",dpi=600,bbox_inches = 'tight',pad_inches =1)


    #FFT
    FFT_Audio_Result ,FFT_Freqs = FFT_Audio(normal_amplitude,sample_rate)
    #DB
    Amplitude_dB  = 10*np.log10(np.abs(FFT_Audio_Result))
    # peaks, _ = find_peaks(np.abs(FFT_Audio_Result), height=10**12)

    #Plot_2_Amplitude (dB) vs frequency (Hz) using logarithmic scale in both axis
    f= plt.figure(figsize=(16,10),dpi=600)
    plt.title("Amplitude_Frequency",fontsize = 26,loc="center")
    plt.xticks(fontsize=14,color='black')
    plt.yticks(fontsize=14,color="black")
    plt.xlabel("Frequency(Hz)",fontsize=18)
    plt.ylabel("Amplitude(dB)",fontsize=18)
    plt.grid(ls='--', lw=1, c='gray',axis="y")
    plt.semilogx(FFT_Freqs[FFT_Freqs>0],Amplitude_dB[FFT_Freqs>0])
    f.savefig(fname="/Users/haoqinghua/Desktop/python/DSP/Amplitude_Frequency.png",dpi=600,bbox_inches = 'tight',pad_inches =1)
    #remove noise
    # remove_noise_audio = Remove_Noise(FFT_Audio_Result,21500)
    # denoise_audio = nf.ifft(remove_noise_audio).real
    # wf.write('/Users/haoqinghua/Downloads/4.wav', sample_rate, denoise_audio)
