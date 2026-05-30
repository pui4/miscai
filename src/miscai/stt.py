import io
try:
    import numpy as np
    from pywhispercpp.model import Model
except:
    raise ImportError("The 'stt' module is required to use this. Install it with 'pip install miscai[stt]'.")

class STT():
    def __init__(self) -> None:
        self.model = Model('base.en')
    
    def transcribe(self, audio_bytes: np.ndarray) -> str:
        segments = self.model.transcribe(audio_bytes)
        
        final = ""
        for segment in segments:
            final += segment.text + "\n"

        return final