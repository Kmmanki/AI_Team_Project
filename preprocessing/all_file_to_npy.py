import os
import numpy as np
import librosa, librosa.display 
import matplotlib.pyplot as plt
import pandas as pd
FIG_SIZE = (8,6)

SR = 11025

def search(dirname):
    x_arr = []
    y_arr = []
    i = 0
    filenames = os.listdir(dirname)
    for filename in filenames:
        y_label = int(filename.split('-')[-2])
        if y_label < 48 or 84 < y_label:
            # print(y_label)
            continue
        full_filename = os.path.join(dirname, filename)
        full_filename = full_filename.replace('\\', '/')
        
        sig, sr = librosa.load(full_filename, sr=SR, offset=1.0)

        # print(sig,sig.shape)

        fft = np.fft.fft(sig)

        # 복소공간 값 절댓갑 취해서, magnitude 구하기
        magnitude = np.abs(fft) 

        # Frequency 값 만들기
        f = np.linspace(0,sr,len(magnitude))

        # 푸리에 변환을 통과한 specturm은 대칭구조로 나와서 high frequency 부분 절반을 날려고 앞쪽 절반만 사용한다.
        left_spectrum = magnitude[:int(len(magnitude)/2)]
        left_f = f[:int(len(magnitude)/2)]

        # print(left_spectrum.shape) #108427
        # 기존의 hz는 너무 크다 줄인다. 
        pitch_index = np.where((left_f > 130.0) & (left_f < 1050.0)) #130 ~ 1050 헤르츠의 index 구함
        # print(left_spectrum.shape) # 9000컬럼
        # print(left_f.shape)

        pitch_freq = left_f[pitch_index] #x축 
        pitch_mag = left_spectrum[pitch_index] #y축 

        def convertFregToPitch(arr):
            return np.round(39.86*np.log10(arr/440.0) + 69.0) #수 많은 소수점 들을 하나로 합치게 해줌. Ex 130.8 130.9 130.10 을 전부 130 => 48로 단일화 즉 값들이 48로 몰링
        convertFregToPitch2 = np.vectorize(convertFregToPitch)

        pitch_freq = convertFregToPitch2(pitch_freq)

        start_index = np.where(pitch_freq>=48)
        # print("start_index:",start_index)
        pitch_freq = pitch_freq[start_index]
        pitch_mag = pitch_mag[start_index]

        # print(pitch_freq)
        # print(pitch_mag)
        # print("pitch_freq.shape:",pitch_freq.shape)
        # print("pitch_mag.shape:",pitch_mag.shape)

        freq_uniq = np.unique(pitch_freq) #여러 미디번호들이 있지만 유니크로 보여주며 유니크 이전엔 48 48 48 48 48 48 48 이런식으로 있을 것이고 해당 인덱스로 주면 mag를 얻는다.
        #mag들의 합을 얻으면 가장 큰 mag를 얻을 수 있다.
        # print(freq_uniq[0]) #대략 미디번호 48 ~ 84까지 있을 거다  # 130 hz ~ 1050 hz 까지 잘랐으니

        tmp_arr = []
        for i in range(len(freq_uniq)):
            # print(freq_uniq[i])
            tmp_avg = np.average(pitch_mag[np.where(pitch_freq == freq_uniq[i])]) # 48을 가진 index 들을 모두 가져와서 avg
            tmp_arr.append(tmp_avg)
        # print(tmp_arr)
        #x 축은 unique 값이지.. x = tmparr y = 가운데 번호
        x_arr.append(tmp_arr)
        y_arr.append(y_label)
        

        # plt.figure(figsize=FIG_SIZE)
        # plt.plot(freq_uniq, tmp_arr)
        # # plt.plot(left_f)
        # plt.xlabel("Frequency")
        # # plt.yscale("log")
        # # plt.xlim([0,2000])
        # # plt.ylim([0,10000])
        # plt.ylabel("Magnitude")
        # plt.title("Power spectrum"+ n[-3:])
        # plt.show()
        # print (y_label)

    return np.array(x_arr), np.array(y_arr), i


x, y, i = search('D:/teamprojectDataset/nsynth-valid/audio')
# y = y.astype('int64')

print(np.unique(y))
print(len(np.unique(y)))
print(i,'개 파일')
np.save('./npy/all_scale_x_11025sr.npy', arr=x)
np.save('./npy/all_scale_y_11025sr.npy', arr=y)