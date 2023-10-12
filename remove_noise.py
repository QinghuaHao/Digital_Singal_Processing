import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf


def fft_transform(audio,sample_rate):
    fft_freqs = nf.fftfreq(len(audio),1/sample_rate)
    fft_audio = nf.fft(audio)
    return fft_audio,fft_freqs
# def remove_noise(fft_audio,b,threshold):
#     fft_audio[(np.abs(fft_audio) < b) & (np.abs(fft_audio) > threshold)] = 0
#     return fft_audio

def remove_noise(fft_freqs,fft_audio,b,threshold):
    fft_audio[fft_freqs < b] = 0
    fft_audio[fft_freqs > threshold] = 0
    return fft_audio


if __name__ == '__main__':
    #import wave
    sample_rate_1cm,audio_1cm= wf.read('./1cm_voice.wav')
    sample_rate_1m,audio_1m = wf.read("./1m_voice.wav")

    fft_audio_1cm , fft_freqs_1cm = fft_transform(audio_1cm,sample_rate_1cm)
    fft_audio_1m ,fft_freqs_1m = fft_transform(audio_1m,sample_rate_1m)

    nonoise_audio_1cm = remove_noise(fft_freqs_1cm,fft_audio_1cm,85,7000)
    nonoise_audio_1m = remove_noise(fft_freqs_1m,fft_audio_1m,85,10000)

    nonoise_audio_singal_1cm = nf.ifft(nonoise_audio_1cm)
    final_audio_singal_1m = nf.ifft(nonoise_audio_1m)

    path_1cm = './1cm_remove_noise.wav'
    path_1m = './1m_remove_noise.wav'
    wf.write(path_1cm,sample_rate_1cm,nonoise_audio_singal_1cm.astype(np.int16))
    wf.write(path_1m,sample_rate_1m,final_audio_singal_1m.astype(np.int16))