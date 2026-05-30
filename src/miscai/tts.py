import math
try:
    import numpy as np
    from pocket_tts import TTSModel
    from scipy.signal import resample_poly
except:
    raise ImportError("The 'tts' module is required to use this. Install it with 'pip install miscai[tts]'.")

class TTS():
    def __init__(self, voice_base_file: str) -> None:
        self.tts_model = TTSModel.load_model()
        self.voice_state = self.tts_model.get_state_for_audio_prompt(
            voice_base_file
        )

    def get_audio(self, text: str):
        audio = self.tts_model.generate_audio(self.voice_state, text)
        
        gcd = math.gcd(self.tts_model.sample_rate, 44100)
        audio_resamp = resample_poly(audio.numpy(), 44100 // gcd, self.tts_model.sample_rate // gcd).astype(np.float32)

        return audio_resamp, 44100
