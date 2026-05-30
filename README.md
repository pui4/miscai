# MiscAI
Tools to make it easier to embed AI related things into your exsisting projects.

## Features
- Ollama local LLM wrapper (with build-in Jina embeddings model)
- Mercury diffusion language model (with the same embeddings as with Ollama, requires a mercury API key)
- Easy tool function creation with both types of language models
- Whisper-cpp speech to text
- Text to speech cloning with pocket-tts (requires a hugging face account)
- Voice activity detection with silero
- Wakeword detection with local wake

## Ollama local LLM
To use the local LLM with Ollama the host has to have Ollama installed already. The embeddings model is ran on the CPU as it is light and performant enough. Then install it into the python project with:
```sh
pip install miscai[llm]
```
Here is an example of how to use it:
```python
from miscai.llm import LLM

llm = LLM(promt="You are helpful assistant.", model="qwen3:4b", convo_file="./convo.json")
print(llm.ask_LLM("Hello!"))
```

## Diffusion Language Model (DLM)
Use of the DLM module is very similar to using the local LLM. You will need to make a Mercury API key to use the wrapper. The embeddings model is ran on the CPU as it is light and performant enough. Then install it into the python project with:
```sh
pip install miscai[dlm]
```
Here is an example of how to use it:
```python
from miscai.dlm import DLM

dlm = DLM(promt="You are helpful assistant.", model="mercury-v2", convo_file="./convo.json", api_key="123abc")
print(dlm.ask_LLM("Hello!"))
```

## Tool calling for Language models
This requires that you have installed one of the language models above. Here is an example of how to use it in a project:
```python
from miscai.tools import ToolLoader

tool_loader = ToolLoader("./tools")
```
Then when you are creating your language model object, create it like this (using the Ollama one for example):
```python
llm = LLM(promt="You are helpful assistant.", model="qwen3:4b", convo_file="./convo.json", tools=tool_loader.get_tools())
```
To create a tool, install the required dependencies to your project and place the python file in the directory specified in the ToolLoader object. Here is an example tool that gets the time with the 'pytz' package:
```python
import pytz
from datetime import datetime

tool = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Gets the current time for a provided time zone.",
        'parameters': {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The time zone as specified in the tz (zoneinfo) library."
                }
            },
            "required": ["timezone"]
        }
    }
}

def get_current_time(timezone: str) -> str:
    tz = pytz.timezone(timezone)
    return str(datetime.now(tz))
```

## Speech to text
The whisper-cpp model is ran on the CPU but is quite performant and accurate, but may use a lot of CPU. The audio bytes are in a numpy array encoded at 16000kHz with 512 byte chunks. Install it into your project with:
```sh
pip install miscai[stt]
```
Here is an example of how to use it:
```python
from miscai.stt import STT

stt = STT()
print(stt.transcribe(audio_bytes=audio_bytes))
```

## Text to speech
You will have to create a hugging face account and accept the eula for using the pocket-tts model. Then create an API key as you will need this later. The model is ran on the CPU for reason mentioned on the model's page. The audio base needs to be encoded with 16000kHz for the best results. The outputed audio is a numpy array. Then install it into the project using:
```sh
pip install miscai[tts]
```
Here is an example of how to use it:
```python
from miscai.tts import TTS

tts = TTS("./voice_base.wav")
audio = tts.get_audio("Hello!")
```

## Voice activity detection
This uses Silero VAD for voice detection and it runs on the CPU due to the model being light weight enough. The audio bytes inputed is a numpy array with 512 byte chunks. For best results the audio should be encoded in 16000kHz. Install it into your project with:
```sh
pip install miscai[vad]
```
Here is an example of how to use it:
```python
from miscai.vad import VAD

vad = VAD(threshold=0.5)
print(vad.is_speech(audio_bytes=audio_bytes))
```

## Wakeword detection
This uses local-wake and so any audio file of any person saying the wakeword works as the wakeword. The audio inputed is a audio stream with 512 byte chunks (best gotten through SoundDevice). For best results the audio for the input and the audio for the reference audio files should be encoded in 16000kHz. Install it into your project with:
```sh
pip install miscai[wake]
```
Here is an example of how to use it:
```python
from miscai.wakeword import WakeWord

wake_word = WakeWord(threshold=0.5, audio_dir="./wakeword")
print("Begining to wait for wakeword")
wake_word.waitForWord(callback=awoke, stream=stream)

def awoke(detection: dict, stream: sd.InputStream):
    print(f"Wake word detected: {detection['wakeword']}")
```

## Final notes
This project is in quite an early stage and could do with some more polish. It is not meant to be used in production but is good for making small prototypes for different ideas that you may have. I made this to make it easier to implement LLMs into my other projects and it grew from there. Changes are welcome as the documention is hastely written and the codes is arguable worse. Thanks for reading and maybe using this. :)