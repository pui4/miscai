try:
    import lwake
except:
    raise ImportError("The 'wake' module is required to use this. Install it with 'pip install miscai[wake]'.")

class WakeWord():
    def __init__(self, threshold: float, audio_dir: str) -> None:
        self.THRESHOLD = threshold
        self.REF_DIR = audio_dir

    def waitForWord(self, callback, stream = None ) -> None:
        lwake.listen(self.REF_DIR, threshold=self.THRESHOLD, method="embedding", callback=callback, stream=stream)