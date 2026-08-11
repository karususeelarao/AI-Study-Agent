import asyncio
import edge_tts


class TTSService:

    async def generate(self, text):

        communicate = edge_tts.Communicate(
            text,
            voice="en-US-AriaNeural",
        )

        await communicate.save("response.mp3")

        return "response.mp3"