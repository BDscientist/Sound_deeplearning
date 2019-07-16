from __future__ import print_function
import freesound  # $ git clone https://github.com/MTG/freesound-pythonf
import mute_constants
import os
import sys

def exec_free_sound(keywords, keyword_download_path):
    api_key = '92wzjQFuwqNWWTC1QWXju3XFkkgEb2Catk3izPAz'
    freesound_client = freesound.FreesoundClient()
    freesound_client.set_token(api_key)
    download = mute_constants.FREESOUND_PATH + keyword_download_path
    for keyword in keywords:
        results_pager = freesound_client.text_search(
            query=keyword,
            # filter="tag:tenuto duration:[1.0 TO 15.0]",
            sort="rating_desc",
            fields="id,name,previews,username"
        )
        os.makedirs(download, exist_ok=True)
        for sound in results_pager:
            print("\t-", sound.name, "by", sound.username)
            try:
                filename = str(sound.id) + '_' + sound.name.replace(u'/', '_') + ".wav"
                if not os.path.exists(download + filename):
                    sound.retrieve_preview(download, filename)
            except Exception as err:
                print("[ Error keyword : " + keyword_download_path + " ]")
                print(err)
        print("Num results:", results_pager.count)
        for page_idx in range(results_pager.count):
            print("----- PAGE  -----" + str(page_idx))
            try:
                results_pager = results_pager.next_page()
            except Exception as err:
                print("[ Error keyword : " + keyword_download_path + " ]")
                print(err)
                break
            try:
                for sound in results_pager:

                    print("\t-", sound.name, "by", sound.username)
                    filename = str(sound.id) + '_' + sound.name.replace(u'/', '_') + ".wav"
                    if not os.path.exists(download + filename):
                        sound.retrieve_preview(download, filename)
            except Exception as err:
                print("[ Error keyword : " + keyword_download_path + " ]")
                print(err)


