try:
    import torch
    import numpy as np
except:
    raise ImportError("The 'vad' module is required to use this. Install it with 'pip install miscai[vad]'.")

class VAD():
    def __init__(self, threshold: float) -> None:
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad') # type: ignore

        self.SAMPLING_RATE = 16000
        self.CHUNK = 512
        self.THRESHOLD = threshold
    
    def is_speech(self, audio_bytes: np.ndarray) -> bool:
        audio_np = audio_bytes.flatten().astype(np.float32)
        audio_tensor = torch.from_numpy(audio_np)
        
        speech_prob = self.model(audio_tensor, self.SAMPLING_RATE).item()

        if speech_prob > self.THRESHOLD:
            return True
        else:
            return False
