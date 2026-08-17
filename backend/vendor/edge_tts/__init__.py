# Minimal stub for edge_tts to allow tests to import TTSService without the
# real dependency. The stub provides a Communicate class with save_sync.
class Communicate:
    def __init__(self, text, voice=None, rate=None):
        self.text = text
        self.voice = voice
        self.rate = rate

    def save_sync(self, path):
        # Write an empty MP3-like file to satisfy callers that expect a file.
        with open(path, "wb") as f:
            f.write(b"")

def Communicate_sync(*args, **kwargs):
    return Communicate(*args, **kwargs)
