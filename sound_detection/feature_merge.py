import glob
import numpy as np

X = np.empty((0, 193))
y = np.empty((1))
groups = np.empty((0, 1))
npz_files = glob.glob('/home/libedev/mute/mute-hero/download/urbansound/UrbanSound8K/audio/Final_DATASET_?.npz')
for fn in npz_files:
    print(fn)
    data = np.load(fn)
    X = np.append(X, data['X'], axis=0)
    y = np.append(y, data['y'], axis=0)
    groups = np.append(groups, data['groups'], axis=0)

    
print(groups[groups>0])

print(X.shape, y.shape)

for r in y:
    if np.sum(r) > 1.5:
        print(r)
np.savez('/home/libedev/mute/mute-hero/download/urbansound/UrbanSound8K/audio/Final_totaldataset.npz', X=X, y=y, groups=groups)