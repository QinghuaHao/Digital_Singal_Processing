
import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt


#compute the frequency of vowel and consonant
def identify_freqs(path,start,end):
    sample_Rate,data = wf.read(path)
    data_1 = data[start:end]
    fft_results = nf.fft(data_1)
    fft_freqs = nf.fftfreq(len(data_1),1/sample_Rate)
    Amplitude_dB =  10*np.log10(np.abs(fft_results))
    return Amplitude_dB,fft_freqs

#compare two different voice
def compare_freqs(ffreq1,ffreq2):
    ffre = []
    for i in ffreq1:
        for y in ffreq2:
            if i == y:
                ffre.append(i)
    return ffre

if __name__ == '__main__':
    path_voice_normal = '/Users/haoqinghua/Desktop/audio/consonant_1.wav'
    path_voice_speed='/Users/haoqinghua/Desktop/audio/sppedconsonant.wav'
    Amplitude_normal,fft_freqs_normal=identify_freqs(path_voice_normal,34000,200000)
    Amplitude_speed ,fft_freqs_speed = identify_freqs(path_voice_speed,34000,200000 )
    print(Amplitude_speed)
    print(Amplitude_normal)
    print(fft_freqs_speed)
    print(fft_freqs_normal)
    plt.gca().semilogx(fft_freqs_normal[fft_freqs_normal>0],Amplitude_normal[fft_freqs_normal>0])
    plt.show()
    plt.gca().semilogx(fft_freqs_speed[fft_freqs_speed>0],Amplitude_speed[fft_freqs_speed>0])
    plt.show()
    path_voice_normal_vowel = '/Users/haoqinghua/Desktop/audio/vowel_1.wav'
    path_voice_speed_vowel = '/Users/haoqinghua/Desktop/audio/sppedvoel.wav'
    Amplitude_normal_vowel,fft_freqs_normal_vowel=identify_freqs(path_voice_normal_vowel,0,120000)
    Amplitude_speed_vowel, fft_freqs_speed_vowel = identify_freqs(path_voice_speed_vowel,0,120000 )
    Freq_normal = fft_freqs_normal[(fft_freqs_normal > 79) & (fft_freqs_normal < 180)]
    Freq_speed =fft_freqs_speed[(fft_freqs_speed > 79) & (fft_freqs_speed < 180)]
    # print(Freq_normal)
    # print(Freq_speed)
    ffr = compare_freqs(Freq_speed,Freq_normal)
    print(ffr[0],ffr[-1])
    Freq_normal_vowel = fft_freqs_normal_vowel[(fft_freqs_normal_vowel > 120) & (fft_freqs_normal_vowel < 640)]
    Freq_speed_vowel = fft_freqs_speed_vowel[(fft_freqs_speed_vowel > 120) & (fft_freqs_speed_vowel < 640)]
    ffr_vowel = compare_freqs(Freq_speed_vowel,Freq_normal_vowel)
    print(ffr_vowel[0],ffr_vowel[-1])
'''
    plt.gca().semilogx(fft_freqs_normal_vowel[fft_freqs_normal_vowel>0],Amplitude_normal_vowel[fft_freqs_normal_vowel>0])
    plt.show()
    plt.gca().semilogx(fft_freqs_speed_vowel[fft_freqs_speed_vowel>0],Amplitude_speed_vowel[fft_freqs_speed_vowel>0])
    plt.show()
    Amplitude_normal = Amplitude_normal[(fft_freqs_normal > 79) & (fft_freqs_normal < 181)]
    print(Amplitude_normal.shape)
    Amplitude_speed = Amplitude_speed[(fft_freqs_speed > 79) & (fft_freqs_speed < 181)]
    print(Amplitude_speed.shape)
    con_index = compare_amplitude(Amplitude_normal,Amplitude_speed,0.1)
    con_index = np.array(con_index)
    print(con_index[0][0])
    print(Amplitude_normal[con_index[0][0]])
    print(fft_freqs_normal[con_index])


    Amplitude_normal_vowel=Amplitude_normal_vowel[(fft_freqs_normal_vowel > 120) & (fft_freqs_normal_vowel < 640)]
    print(Amplitude_normal_vowel.shape)
    Amplitude_speed_vowel = Amplitude_speed_vowel[(fft_freqs_speed_vowel > 120) & (fft_freqs_speed_vowel < 640)]
    print(Amplitude_speed_vowel.shape)
    print('/')

    # con_index = compare_amplitude(Amplitude_normal,Amplitude_speed,1)
    # vow_index = compare_amplitude(Amplitude_normal_vowel,Amplitude_speed_vowel,1)
'''
