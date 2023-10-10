import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
#傅立叶变换
def fft_transform(audio,sample_rate):
    fft_freqs = nf.fftfreq(len(audio),1/sample_rate)
    fft_audio = nf.fft(audio)
    return fft_audio,fft_freqs
def remove_noise(fft_audio,threshold):
    fft_audio[np.abs(fft_audio)>threshold]=0
    return fft_audio
def ifft_transform(fft_audio):
    denoise_audio = nf.ifft(fft_audio).real
    return denoise_audio


if __name__ == '__main__':
    #import wave
    sample_rate_1cm,audio_1cm= wf.read('./1cm_voice.wav')
    sample_rate_1m,audio_1m = wf.read("./1m_voice.wav")
    #进行傅立叶变换
    fft_audio_1cm , fft_freqs_1cm = fft_transform(audio_1cm,sample_rate_1cm)
    fft_audio_1m ,fft_freqs_1m = fft_transform(audio_1m,sample_rate_1m)
    #去除噪声
    nonoise_audio_1cm = remove_noise(fft_audio_1cm,2000)
    nonoise_audio_1m = remove_noise(fft_audio_1m,2000)
    #傅立叶逆向变换
    nonoise_audio_singal_1cm = ifft_transform(nonoise_audio_1cm)
    final_audio_singal_1m = ifft_transform(nonoise_audio_1m)
    #保存文件
    path_1cm = '/Users/haoqinghua/Desktop/python/1.wav'
    path_1m = '/Users/haoqinghua/Desktop/python/2.wav'
    wf.write(path_1cm,sample_rate_1cm,nonoise_audio_singal_1cm)
    wf.write(path_1m,sample_rate_1m,final_audio_singal_1m)