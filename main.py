import os
import whisper
import ssl
import uuid

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context
base_model = whisper.load_model("small")



def load_example():
    return inference('files/OSR_us_000_0010_8k.wav')

    
def inference(audio):
    """
    Gets audio speech recog through whisper 

    Parameters
    ----------
    audio: audio-file-name

    Returns
    -------
    text (str) : Text out of audio
    """

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
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
    filename='files/'+str(uuid.uuid4().hex)+'.wav'
    with open(filename, mode='bx') as f:
        f.write(wave_bytes)
    text = inference(filename )
    os.remove(filename)
    return text

with open('files/OSR_us_000_0010_8k.wav',mode='rb') as f:
    a= f.read()

print(infer_wave_byte(a ))