from source import mute_constants
import youtube_dl
import time

from pydub import AudioSegment
import os

if __name__ == '__main__':
    # path = mute_constants.DOWNLOAD_PATH
    # video_url = 'http://www.youtube.com/watch?v=--PJHxphWEs'
    # downloadYouTube(video_url, path)

    milli_sec = int(round(time.time() * 1000))
    path = mute_constants.TEMP_PATH + 'Video_' + str(milli_sec) + '.%(ext)s'
    ydl_opts = {
        'outtmpl': path,
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info("https://www.youtube.com/watch?v=-7J4109yM7w", download=True)
        filename = ydl.prepare_filename(info)
        fname, ext = os.path.splitext(filename)
        ext = ext.replace(".", "")
        print("[ filename : " + fname + " , ext :  " + ext + " ]")
        if ext == "webm" or ext == "m4a":
            ext = "mp3"
            filename = fname + "." + ext
        sound_file = AudioSegment.from_file(filename, format=ext)

    # sound_file = AudioSegment.from_mp3("/Users/alex/works/workspaces/work_mute/mute-hero/download/temp/Video_1560240660758.mp3")
    # new_file = sound_file[420 * 1000: 430 * 1000]
    # new_file.export("/Users/alex/works/workspaces/work_mute/mute-hero/download/out.wav", format="wav")

