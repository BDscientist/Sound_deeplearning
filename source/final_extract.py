import numpy  # 수치 연산에 이용
import librosa # 음원 파일을 읽고 분석하는 데 이용
import os # 디렉토리 생성 등 시스템 관련 작업
import os.path # 특정 경로가 존재하는지 파악하기 위해 필요
import sys
import io
import sklearn
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')


sequence_length = 251
feature_dimension = 513

def extract_spectrum(part):
    sample_files = open('C:/project1/'+part + '_samples.txt').read().strip().split('\n') # 샘플 목록을 읽어옵니다.
    if part == 'train': # 'train'인 경우에는 평균과 표준편차를 구해야 합니다.
        data_sum = numpy.zeros((sequence_length, feature_dimension)) # 합계를 저장할 변수를 만듭니다.
        data_squared_sum = numpy.zeros((sequence_length, feature_dimension)) # 제곱의 합을 저장할 변수입니다.
    if not os.path.exists('C:/project1'+part+'/spectrum/'): # 'spectrum' 디렉토리가 존재하지 않으면 만들어 줍니다.
        os.mkdir('C:/project1/'+part+'/spectrum/')
    for f in sample_files:
        print('%d/%d: %s'%(sample_files.index(f), len(sample_files), f)) # 현재 진행상황을 출력합니다.
        y, sr = librosa.load('C:/project1/'+part+'/audio/'+f+'.wav', sr=16000) # librosa를 이용해 샘플 파일을 읽습니다.
        D = librosa.stft(y, n_fft=1024, hop_length=256).T # short-time Fourier transform을 합니다.
        mag, phase = librosa.magphase(D) # phase 정보를 제외하고, 세기만 얻습니다.
        S = numpy.log(1 + mag * 1000) # 로그형태로 변환합니다.
        if part == 'train': # 'train'인 경우 합계와 제곱의 합을 누적합니다.
            data_sum += S
            data_squared_sum += S ** 2
        numpy.save('C:/project1/'+part+'/spectrum/'+f+'.npy', S) # 현재 샘플의 스펙트럼을 저장합니다.
    if part == 'train': # 모든 파일의 변환이 끝난 후에, 'train'인 경우 평균과 표준편차를 저장합니다.
        data_mean = data_sum / len(sample_files)
        data_std = (data_squared_sum / len(sample_files) - data_mean ** 2) ** 0.5
        numpy.save('C:/project1/'+part+'/spectrum/data_mean.npy', data_mean)
        numpy.save('C:/project1/'+part+'/spectrum/data_std.npy', data_std)
        print("finish!!!")





if __name__ == '__main__':

    for part in ['train','test','valid']:
        extract_spectrum(part)







#stft 를 가지고 변환 후 log ---> pca 를 통해 차원을 축소할것임


#이 다음으로 가우시안 믹스쳐 모델을 사용해야 하는지? cnn 이나 rnn model을 사용해야하는지? 고민
# cnn 구조로 모델링하고 EM 알고리즘을 찾아봄. MIXTURE GUASIAN MODEL도 하나의 방법
#DATA 패턴인식 네트워크 구조 : 데이터를 전처리하는데 있어서 같은 시간
