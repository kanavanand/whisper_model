## Whisper 

Whisper is an automatic speech recognition model trained on 680,000 hours of multilingual data collected from the web. As per OpenAI, this model is robust to accents, background noise and technical language. In addition, it supports 99 different languages’ transcription and translation from those languages into English.. In this Daisi I have used the "small" model, but soon I'll be updating the other 4 variants of this model as well.

##### Use daisi-api to make calls

Step1- Load model
```
import pydaisi as pyd
whisper_model_gpu = pyd.Daisi("kanav/Whisper Model-GPU")
```
Step2- Make predictions <br>
Predictions can be made sending byte audio object or audio array (in float format ) <br>
you can download sample audio from here - https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav

filename - OSR_us_000_0010_8k.wav

From audio array(single dimensional)
```
audio = whisper.load_audio('files/OSR_us_000_0010_8k.wav')
whisper_model_gpu.inference(audio).value
[Output]>>> The birch canoe slid on the smooth planks. Glued the sheet to the dark blue background. It is easy to tell the depth of a well. These days a chicken leg is a rare dish. Rice is often served in round bowls. The juice of lemons makes fine punch. The box was thrown beside the parked truck. The hogs were fed chopped corn and garbage.
```
From byte like object

```
### loading bytes data
with open('files/OSR_us_000_0010_8k.wav',mode='rb') as f:
        wave_bytes= f.read()
whisper_model_gpu.infer_wave_byte(wave_bytes).value
[Output]>>> The birch canoe slid on the smooth planks. Glued the sheet to the dark blue background. It is easy to tell the depth of a well. These days a chicken leg is a rare dish. Rice is often served in round bowls. The juice of lemons makes fine punch. The box was thrown beside the parked truck. The hogs were fed chopped corn and garbage.
```

### Web-UI > https://app.daisi.io/daisies/kanav/Whisper-WebUI/app