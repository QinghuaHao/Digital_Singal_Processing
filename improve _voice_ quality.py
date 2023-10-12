import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf


def fft_transform(audio,sample_rate):
    fft_freqs = nf.fftfreq(len(audio),1/sample_rate)
    fft_audio = nf.fft(audio)
    return fft_audio,fft_freqs

def improve_voice_quality(fft_audio,star_freq,end_freq,a):
    index = np.where((star_freq<np.abs(fft_audio))&(np.abs(fft_audio)<end_freq))
    change_voice = fft_audio[index]*a
    fft_audio[index] = change_voice
    return fft_audio


if __name__ == '__main__':
    #import wave
    sample_rate_1cm,audio_1cm= wf.read('./1cm_voice.wav')
    sample_rate_1m,audio_1m = wf.read("./1m_voice.wav")

    fft_audio_1cm , fft_freqs_1cm = fft_transform(audio_1cm,sample_rate_1cm)
    fft_audio_1m ,fft_freqs_1m = fft_transform(audio_1m,sample_rate_1m)

    voice_1cm = improve_voice_quality(fft_audio_1cm,100,8000,5)
    voice_1m = improve_voice_quality(fft_audio_1m,100,8000,8)

    nonoise_audio_singal_1cm = nf.ifft(voice_1cm)
    final_audio_singal_1m = nf.ifft(voice_1m)

    path_1cm = './1cm_improve_voice.wav'
    path_1m = './1m_improve_voice.wav'
    wf.write(path_1cm,sample_rate_1cm,nonoise_audio_singal_1cm.astype(np.int16))
    wf.write(path_1m,sample_rate_1m,final_audio_singal_1m.astype(np.int16))