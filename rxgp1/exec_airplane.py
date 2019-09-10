import sys
sys.path.append('..')

from rxgp1 import exec_freesound

if __name__ == '__main__':
    keywords = ["airplane", "aircraft"]
    exec_freesound.exec_free_sound(keywords, "airplane")