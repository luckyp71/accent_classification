from io import BytesIO
import librosa
import ffmpeg
import yt_dlp


def get_direct_audio_url(youtube_url):
    ydl_opts = {'quiet': True, 'format': 'bestaudio'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']


def extract_audio_from_url(url, target_sr=16000):
    out, _ = (
        ffmpeg
        .input(url)
        .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=target_sr)
        .run(capture_stdout=True, capture_stderr=True)
    )
    audio, sr = librosa.load(BytesIO(out), sr=target_sr)
    return audio, sr