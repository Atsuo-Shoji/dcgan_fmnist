# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pickle

def do_label_smoothing(label, epsilon=0.005):
    #ラベル平滑化
    #クラス分類タスクの正則化手法の1つ
    
    if epsilon > 0:
        class_num = label.shape[1]      
        label_smoothed = np.where(label == 1, (1.0-epsilon), epsilon/(class_num-1))
    else:
        label_smoothed = label
          
    return label_smoothed

def convert_scale_img_data(data):
    #画像データの画素値を[-1, 1]の間へとスケール変換
    
    min_value = data.min()
    max_value = data.max()
    
    #数値の最小値と最大値を見て、float画像か整数画像か判断。
    #dtypeでの判定は行わない。整数画像（のつもり）でdtypeがfloat、というのはよくある話。
    if 0<=min_value and max_value<=1.0:
        #[0, 1]の値域に既に正規化されているfloat画像と認識
        data = data * 2.0 - 1.0
    elif 0<=min_value and max_value<=255:
        #[0, 255]の値域
        data = (data.astype(np.float32) - 127.5) / 127.5
    else:
        #その他は何もしない。
        pass    
    
    return data
    
def normalize_data(data):
    #正規化
    
    min_value = data.min()
    max_value = data.max()
        
    #数値の最小値と最大値を見て、float画像か整数画像か判断。
    #dtypeでの判定は行わない。整数画像（のつもり）でdtypeがfloat、というのはよくある話。
    if 0<=min_value and max_value<=1.0:
        #[0, 1]の値域に既に正規化されているfloat画像と認識
        #何もしない
        pass
    elif 0<=min_value and max_value<=255:
        #[0, 255]の値域
        data = (data - min_value) / (max_value - min_value)
        data = data.astype('float32')
    else:
        #その他は何もしない。
        pass
    
    return data

def show_images(imgs, img_shape, rows, cols, channels_first=True, size_unit_inch=1):
    #複数の画像を一覧表示する
    
    #チャンネル数を取得
    if channels_first==True:
        C, H, W = img_shape
    else:
        H, W, C = img_shape
        
    num_imgs = imgs.shape[0]
    
    #imshowの仕様で、チャンネルが1の場合は、そのチャンネルのaxisをつぶさないといけない。
    if C==1:
        imgs = imgs.reshape(-1, H, W)            
            
    fig, axs = plt.subplots(rows, cols, figsize=(cols*size_unit_inch, rows*size_unit_inch), sharex=True, sharey=True) 
    curr_idx_imgs = 0
    for i in range(rows):
        for j in range(cols):
            if curr_idx_imgs < num_imgs:
                if C==1:
                    axs[i, j].imshow(imgs[curr_idx_imgs], cmap='gray')
                else:
                    axs[i, j].imshow(imgs[curr_idx_imgs])                
            axs[i, j].axis('off')    
            curr_idx_imgs += 1
    plt.show()
    
def read_pickle_file(file_path):
    #指定されたパスのpickleファイルを読み込む。
    
    with open(file_path, "rb") as fo:
        obj = pickle.load(fo)
        
    return obj
        
def save_pickle_file(obj, file_path):
    #指定されたオブジェクトを指定されたパスのpickleファイルとして書き込む。
    
    with open(file_path, 'wb') as fo:
        pickle.dump(obj , fo) 
    
def timedelta_HMS_string(td):
    #TimeDeltaオブジェクトを文字列表現にする。
    #xx hours xx minutes xx seconds　というように
    
    hours = td.seconds//3600
    remainder_secs = td.seconds%3600
    minutes = remainder_secs//60
    seconds = remainder_secs%60
    
    HMS_string = str(hours) + " hours " + str(minutes) + " minutes " + str(seconds) + " seconds"
    
    return HMS_string