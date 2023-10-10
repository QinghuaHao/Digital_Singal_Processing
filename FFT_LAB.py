import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

# Find the index closest to the frequency given
def Find_idx_of_freq(freqs,target_freq): #完成
    abs_diff = np.abs(freqs - target_freq)
    return np.argmin(abs_diff)

# Find the index of peak in amplitude
def Find_one_peak_idx(start_freq_idx,end_freq_idx,amplitude):
    return np.argmax(amplitude[start_freq_idx:end_freq_idx])+start_freq_idx

# according to the frequencies given, split the frequency spectrum into segments
def Find_peaks_idx(separation_point_arr,freqs,amplitude):
    peaks_idx = []
    for i in range(len(separation_point_arr)-1):
        start_freq_idx = Find_idx_of_freq(freqs,separation_point_arr[i])
        end_freq_idx = Find_idx_of_freq(freqs,separation_point_arr[i+1])
        peaks_idx.append(Find_one_peak_idx(start_freq_idx,end_freq_idx,amplitude))
    return np.asarray(peaks_idx)
def Generate_rectangle(x_start_freq,x_end_freq,height,color):
    # 这个时候，x轴已经是频率了，输入多少就是多少
    x_start = x_start_freq
    y_start = -30
    width = x_end_freq - x_start

    height = height*2
    return Rectangle((x_start, y_start), width, height, fill=False, color=color)


if __name__ == '__main__':
    sample_rate, audio = wf.read("./1cm_voice.wav")  # add address of wav profile
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
    Amplitude_dB = 10*np.log10(np.abs(FFT_Audio_Result))

    # Here you can find peaks in segments you given,
    # for example, now the segments are 100Hz-200Hz, 200Hz-400Hz, 400Hz-900Hz
    test_sept = [100,200,400,900]
    peaks_idx = Find_peaks_idx(test_sept,FFT_Freqs[FFT_Freqs>0],Amplitude_dB[FFT_Freqs>0])

    # Here you can plot blocks to mark the frequency range
    # for example, now the blocks are from 100Hz-500Hz, 65Hz-1500Hz
    rectangles = []
    rectangles.append(Generate_rectangle(100,500,max(Amplitude_dB),'green'))
    rectangles.append(Generate_rectangle(65,1500,max(Amplitude_dB),'orange'))


    #Plot_2_Amplitude (dB) vs frequency (Hz) using logarithmic scale in both axis
    f= plt.figure(figsize=(16,10),dpi=600)
    plt.title("Amplitude_Frequency",fontsize = 26,loc="center")
    plt.xticks(fontsize=14,color='black')
    plt.yticks(fontsize=14,color="black")
    plt.xlabel("Frequency(Hz)",fontsize=18)
    plt.ylabel("Amplitude(dB)",fontsize=18)
    plt.grid(ls='--', lw=1, c='gray',axis="y")
    plt.semilogx(FFT_Freqs[FFT_Freqs>0],Amplitude_dB[FFT_Freqs>0])
    #mark vowels peaks
    plt.scatter(FFT_Freqs[peaks_idx], Amplitude_dB[peaks_idx], s=400, marker='o', facecolors='none', edgecolors='red', zorder=3)
    # Mark the frequency range contains the consonants/ harmonics
    for rect in rectangles:
        plt.gca().add_patch(rect)
    f.savefig(fname="./Amplitude_Frequency.png",dpi=600,bbox_inches = 'tight',pad_inches =1)
    #remove noise
    # remove_noise_audio = Remove_Noise(FFT_Audio_Result,21500)
    # denoise_audio = nf.ifft(remove_noise_audio).real
    # wf.write('/Users/haoqinghua/Downloads/4.wav', sample_rate, denoise_audio)
