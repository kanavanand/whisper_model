import os
import whisper
import ssl
import io
import librosa
import numpy as np
import soundfile as sf

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
ssl._create_default_https_context = ssl._create_unverified_context
base_model = whisper.load_model("small")



def load_example():
    """
    Returns
        -------
    text (str) : Text out of sample audio audio
    """
    with open('files/OSR_us_000_0010_8k.wav',mode='rb') as f:
        a= f.read()
    return infer_wave_byte(a)


def __version__():
    return "0.1"


def inference(audio):
    """
    Gets audio speech recog through whisper 

    Parameters
    ----------
    audio: audio array with samplerate = 16k

    Returns
    -------
    text (str) : Text out of audio
    """

    # load audio and pad/trim it to fit 30 seconds
    # audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(base_model.device)

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(base_model, mel, options)
    return result.text

def infer_wave_byte( wave_bytes ):
    """
        Gets audio speech recog through whisper 

        Parameters
        ----------
        wave_bytes: byte format of audio

            with open('files/OSR_us_000_0010_8k.wav',mode='rb') as f:
                wave_byte = f.read()
            you can use above code to read file in byte format and send it to the daisi endpoint
        Returns
        -------
        text (str) : Text out of audio
    """
    data, samplerate = sf.read(io.BytesIO(wave_bytes))
    y_8k = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
    y_8k = y_8k.astype(np.float32)
    text = inference(y_8k )
    return text

