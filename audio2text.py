from openai import OpenAI
import os

def audio2text(audio_file, client):

    # 오디오 파일을 읽어서, 위스퍼를 사용한 변환
    audio_file = open(audio_file, "rb")

    # 결과 반환
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        language="ko",
        response_format="text",
    )

    return transcript
