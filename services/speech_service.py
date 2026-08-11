import whisper


class SpeechService:

    def __init__(self):
        self.model = whisper.load_model("base")

    def transcribe(self, audio_path):

        result = self.model.transcribe(audio_path)

        return result["text"]