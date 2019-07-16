import os
import file_util, logger, mute_constants
import time
import youtube_dl
from pydub import AudioSegment


def download (download_path, csv_name):
    download_path = download_path
    csv = file_util.csv_util(csv_name)
    if csv.is_csv_exists():
        for dic in csv.get_csv_object():
            query = dic[0]
            start = dic[1]
            end = dic[2]
            category_key = dic[3].replace("\"", "").strip()
            category_val = csv.get_category(category_key)
            if category_val is not None:
                category_path = download_path + category_val
                logger.debug("create file [ " + query + " / " + start + " / " + end + " / " + category_key + " ]")
                print("[ " + query + " / " + start + " / " + end + " / " + category_key + " ]")
                try:
                    os.makedirs(category_path, exist_ok=True)
                    milli_sec = int(round(time.time() * 1000))
                    temp = mute_constants.TEMP_PATH + 'Video_' + str(milli_sec) + '.%(ext)s'
                    ydl_opts = {
                        'outtmpl': temp,
                        'format': 'bestaudio/best',
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                            'preferredquality': '192',
                        }],
                    }
                    url = "https://www.youtube.com/watch?v=" + query
                    print(url)
                    logger.debug("[ download from url " + url + "]")
                    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                        filename = ydl.prepare_filename(info)
                        fname, ext = os.path.splitext(filename)
                        ext = ext.replace(".", "")
                        print("[ filename : " + fname + " , ext :  " + ext + " ]")
                        logger.debug("[ convert to filename : " + fname + " , ext :  " + ext + " ]")
                        if ext == "webm" or ext == "m4a":
                            ext = "mp3"
                            filename = fname + "." + ext
                        sound_file = AudioSegment.from_file(filename, format=ext)

                        new_file = sound_file[float(start) * 1000: float(end) * 1000]
                        new_file_name = category_path + "/" + query + ".wav"
                        new_file.export(new_file_name, format="wav")
                        logger.debug("[ convert to filename : " + fname + " , ext :  " + ext + " ]")
                        # break
                except Exception as err:
                    logger.error("error to [ " + query + " / " + start + " / " + end + " / " + category_key + " ]")
                    logger.error(err)
    csv.close()


if __name__ == '__main__':
    path = mute_constants.TEST_PATH
    download(path, 'eval_segments.csv')


