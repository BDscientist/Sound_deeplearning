import glob
import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(".."))
from rxgp1 import mute_constants

if __name__ == '__main__':
    X = np.empty((0, 193))
    y = np.empty((0, 10))
    groups = np.empty((0, 1))
    npz_files = glob.glob(mute_constants.BASE_PATH + "data/train/each_npz/urban_sound_*.npz")
    for fn in npz_files:
        print(fn)
        data = np.load(fn)
        X = np.append(X, data['X'], axis=0)
        y = np.append(y, data['y'], axis=0)
        groups = np.append(groups, data['groups'], axis=0)

    print(groups[groups > 0])

    print(X.shape, y.shape)
    for r in y:
        if np.sum(r) > 1.5:
            print(r)
    np.savez(mute_constants.BASE_PATH + "data/train/total_npz/urban_sound", X=X, y=y, groups=groups)