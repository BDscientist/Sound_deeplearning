import sys
sys.path.append('..')
from NEW_Data_preprocessing import Convert_Numpy

   
if __name__ == '__main__':

    for part in ['new_train','new_test']:

        extract_custom_mfcc(part)