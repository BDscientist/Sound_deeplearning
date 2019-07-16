import os
import os.path
import sys


def move_file(part):
    label_list = open('/home/libedev/mute/mute-hero/download/dataset/sample.txt').read().strip().split('\n')
    files =[]
    labels =[]
    all_files = os.listdir('/home/libedev/mute/mute-hero/download/dataset/'+part)
    
    for f in all_files:
        if f[-4:] =='.wav':
            files.append(f[:-4])
            label = f.split('_')[1]
            labels.append(label_list.index(label))
        
        file_out = open('/home/libedev/mute/mute-hero/download/dataset/'+part+'_samples.txt','w')
        
        for f in files:
            file_out.write(f+'\n')
        file_out.close()
        
        label_out = open('/home/libedev/mute/mute-hero/download/dataset/'+part+'_samples.txt','w')
        
        for l in labels:
            label_out.write(str(l)+'\n')
        label_out.close()

        
if __name__ == '__main__' :
    
    
    move_file('new_train')
    move_file('new_test')