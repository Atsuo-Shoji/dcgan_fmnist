# -*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential #モデルの実体　Sequential記法
from tensorflow.keras.layers import (Activation, BatchNormalization, Dense, Dropout) #モデルを構成する各layerの定義のため
from tensorflow.keras.layers import (Conv2D, Flatten, Conv2DTranspose, Reshape) #モデルを構成する各layerの定義のため　画像認識と生成用
from tensorflow.keras.optimizers import Adam #オプティマイザーにAdamを使用するため
from tensorflow.keras.layers import LeakyReLU #活性化関数にLeakyReLUを使用するため
from tensorflow.keras.models import load_model #generatorとdiscriminatorの保存ファイルをloadするため

'''
from keras.models import Sequential #モデルの実体　Sequential記法
from keras.layers import (Activation, BatchNormalization, Dense, Dropout) #モデルを構成する各layerの定義のため
from keras.layers import (Conv2D, Flatten, Conv2DTranspose, Reshape) #モデルを構成する各layerの定義のため　画像認識と生成用
from keras.optimizers import Adam #オプティマイザーにAdamを使用するため
from keras.layers.advanced_activations import LeakyReLU #活性化関数にLeakyReLUを使用するため
from keras.models import load_model #generatorとdiscriminatorの保存ファイルをloadするため
'''

import numpy as np
from datetime import datetime
from common.tools import *

class dcgan_fmnist_kr():
    
    #訓練時に生成するサンプルイメージの枚数
    _NUM_SAMPLE_IMGS_CONST = 32
    
    def __init__(self, name, z_dim=100, img_shape=(28, 28, 1)):
        
        self._name = name
        self._z_dim = z_dim
        self._img_shape = img_shape
        
        #訓練時に生成するサンプルイメージの元になる固定の潜在変数z
        self._z_fixed_for_sample = np.random.normal(0, 1, (dcgan_fmnist_kr._NUM_SAMPLE_IMGS_CONST, self._z_dim))
    
        #generatorの実体の生成
        self._generator = self._prepare_layers_generator(z_dim, img_shape)
        #discriminatorの実体の生成
        self._discriminator = self._prepare_layers_discriminator(img_shape)
        
        #discriminatorを使用可能に
        self._discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8), 
                                     metrics=['accuracy'])
        #combied modelの__gan_combined（コンテナ）内では、generatorのみ訓練し、discriminatorは訓練対象外であるので以下の設定をする。
        #ただし設定はcombied modelの__gan_combined（コンテナ）のcompieの前にやる必要がある。
        self._discriminator.trainable = False
        #↑既にdiscriminatorはcompile済であり、この設定変更が影響するのはこれから生成するcombied modelのコンテナに対する訓練時のみ。
        #つまりdiscriminator単体に対する訓練はちゃんと行われる。
        
        #combied modelの__gan_combined（コンテナ）の実体の生成
        self._gan_combined = self._prepare_combined_models_gan(self._generator, self._discriminator)
        #_gan_combinedを使用可能に
        self._gan_combined.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)) 
        
    def _prepare_layers_generator(self, z_dim, img_shape):
        
        ##########カスタマイズ可能箇所##########
        ####オリジナルでは、Fashion-MNISTにチューニングしてあります。####
        ####以下の箇所を必要に応じてカスタマイズしてください。####
        
        #generatorの実体を準備して返す。layersをつなげた状態で返す。compileはしない。
        
        #generatorの実体はSequantialのインスタンス。Sequential記法。
        model_g = Sequential()
        
        if img_shape==(28, 28, 1):
            
            #28x28　1チャンネルの画像の場合

            length_before_first_deconv2d = np.ceil(img_shape[0] / 4).astype(np.int)
            ch_generated_img = img_shape[2]

            #第1層　全結合　潜在変数z(N, z_dim)→(N, 7, 7, 256)に
            #Affine Layer (N, z_dim)→(N, 7*7*256=12544)
            model_g.add(Dense(input_dim=z_dim, units=length_before_first_deconv2d*length_before_first_deconv2d*256, 
                              use_bias=True, activation=None, kernel_initializer='he_normal', bias_initializer='zeros'))
            #(N, 7*7*256=12544)→(N, 7, 7, 256)
            model_g.add(Reshape((length_before_first_deconv2d, length_before_first_deconv2d, 256)))
            
            #第2層　転置畳み込み　(N, 7, 7, 256)→(N, 14, 14, 128)
            #転置畳み込みlayer
            model_g.add(Conv2DTranspose(filters=128, kernel_size=(3,3), strides=(2,2), padding='same', data_format='channels_last',
                            use_bias=True, activation=None, kernel_initializer='he_normal', bias_initializer='zeros'))
            #バッチ正規化layer
            model_g.add(BatchNormalization(axis=3)) 
            #活性化関数layer　LeakyReLU
            model_g.add(LeakyReLU(alpha=0.01))
            
            #第3層　転置畳み込み　(N, 14, 14, 128)→(N, 14, 14, 64)
            #転置畳み込みlayer
            model_g.add(Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', data_format='channels_last',
                            use_bias=True, activation=None, kernel_initializer='he_normal', bias_initializer='zeros'))
            #バッチ正規化layer
            model_g.add(BatchNormalization(axis=3))
            #活性化関数layer　LeakyReLU
            model_g.add(LeakyReLU(alpha=0.01))
                        
            #第4層（出力）　転置畳み込み　(N, 14, 14, 64)→(N, 28, 28, 1)
            #転置畳み込みlayer　活性化関数はtanh
            model_g.add(Conv2DTranspose(filters=ch_generated_img, kernel_size=(3,3), strides=(2,2), padding='same', data_format='channels_last',
                            use_bias=True, activation='tanh', kernel_initializer='glorot_normal', bias_initializer='zeros')) 
        
        else:
            raise ValueError("対象外のimg_shapeです。指定されたimg_shapeに対応させるにはカスタマイズが必要です。")

        return model_g
    
        ##########カスタマイズ可能箇所　終わり##########
    
    def _prepare_layers_discriminator(self, img_shape):
        
        ##########カスタマイズ可能箇所##########
        ####オリジナルでは、Fashion-MNISTにチューニングしてあります。####
        ####以下の箇所を必要に応じてカスタマイズしてください。####        
        
        #discriminatorの実体を準備して返す。layersをつなげた状態で返す。compileはしない。
        
        #discriminatorの実体はSequantialのインスタンス。Sequential記法。
        model_d = Sequential()
        
        if img_shape==(28, 28, 1):
            
            #28x28　1チャンネルの画像の場合

            #第1層　畳み込み　(N, 28, 28, 1)→(N, 14, 14, 32)
            #畳み込みlayer
            model_d.add(Conv2D(input_shape=img_shape, filters=32, kernel_size=(3,3), strides=(2,2), padding='same', 
                              data_format='channels_last', use_bias=True, activation=None, 
                              kernel_initializer='he_normal', bias_initializer='zeros'))
            model_d.add(LeakyReLU(alpha=0.01))
            
            #第2層　畳み込み　(N, 14, 14, 32)→(N, 7, 7, 64)
            #畳み込みlayer
            model_d.add(Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='same', data_format='channels_last',
                            use_bias=True, activation=None, kernel_initializer='he_normal', bias_initializer='zeros'))
            #活性化関数layer　LeakyReLU
            model_d.add(LeakyReLU(alpha=0.01))
            
            #第3層　畳み込み　(N, 7, 7, 64)→(N, 4, 4, 128)
            #畳み込みlayer
            model_d.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='same', data_format='channels_last',
                            use_bias=True, activation=None, kernel_initializer='he_normal', bias_initializer='zeros'))
            #活性化関数layer　LeakyReLU
            model_d.add(LeakyReLU(alpha=0.01))

            #第4層（出力）　2値分類の確率
            #画像Tensorを2次元にするlayer　(N, 4, 4, 128)→(N, 4*4*128)
            model_d.add(Flatten()) #(N, 4*4*128)
            #全結合&シグモイドlayer
            model_d.add(Dense(units=1, 
                              use_bias=True, activation='sigmoid', kernel_initializer='glorot_normal', bias_initializer='zeros'))

        else:
            raise ValueError("対象外のimg_shapeです。指定されたimg_shapeに対応させるにはカスタマイズが必要です。")
        
        return model_d
    
        ##########カスタマイズ可能箇所　終わり##########
    
    def _prepare_combined_models_gan(self, generator, discriminator):
        #generatorとdiscriminatorを結合させたcombined modelを返す。modelをつなげた状態で返す。compileはしない。
        
        model_gan = Sequential()
        model_gan.add(generator)
        model_gan.add(discriminator)
        
        return model_gan
    
    def generate(self, z):
        #画像を生成する。
        #z：潜在変数　(生成画像枚数, z_dim)
        
        #投入されたzは1個だけ、ということも考えられる　z.reshape(-1, self._z_dim)だけでもいいが、常に余計な演算がなされる
        if z.ndim==1:
            z = z.reshape(1, self._z_dim)
        
        #generatorによる画像生成
        imgs_generated = self._generator.predict(z, batch_size=256)
        #このimgs_generatedはtanhの出力であり、値域は[-1,1]。しかしそれでは画像データにならないので、値域を[0,1]にする。
        imgs_generated = 0.5 * (imgs_generated + 1.0)
        
        return imgs_generated

    def discriminate(self, imgs):
        #モデル性能計測用。
        #画像データが本物である確率を返す。
        #imgs：判定する画像。(判定画像枚数, 28, 28, 1)。値域は[0, 1]か「0, 255」であること。
        
        #投入された画像は1枚だけ、ということも考えられる　imgs.reshape(-1, *self._img_shape)だけでもいいが、常に余計な演算がなされる
        if imgs.ndim==3:
            imgs = imgs.reshape(1, *self._img_shape)
        
        #generatorの生成データはtanhの出力であり値域は[-1,1]。discriminatorに吸わせるデータもこの値域に合わせるようにスケール変換。
        imgs = convert_scale_img_data(imgs)
        #確率
        probs = self._discriminator.predict(imgs, batch_size=256)
        
        return probs
    
    def evaluate(self, imgs, labels):
        #モデル性能計測用。
        #discriminatorのlossとaccuracy、generatorのlossを返す。
        #imgsとlabels：discriminatorの評価をするために使用するためのデータ。画像群と、それら1枚1枚に対応する本物/偽物ラベル。
        #imgsは、shapeは(画像枚数, 28, 28, 1)、値域は[0, 1]か「0, 255」であること。
        
        #投入された画像は1枚だけ、ということも考えられる　imgs.reshape(-1, *self._img_shape)だけでもいいが、常に余計な演算がなされる
        if imgs.ndim==3:
            imgs = imgs.reshape(1, *self._img_shape)
        
        #discriminator
        #generatorの生成データはtanhの出力であり値域は[-1,1]。discriminatorに吸わせるデータもこの値域に合わせるようにスケール変換。
        imgs = convert_scale_img_data(imgs)
        d_score = self._discriminator.evaluate(x=imgs, y=labels, batch_size=256, verbose=False)
        d_loss, d_accuracy = d_score[0], d_score[1] 
        
        #generator
        #そうする必要は無いが、zの個数は、labelsの個数とする。
        num_data = labels.shape[0]
        z = np.random.normal(0, 1, (num_data, self._z_dim))
        labels_real = np.ones((num_data, 1)) #正解ラベルは「本物」であることに注意。
        g_loss = self._gan_combined.evaluate(x=z, y=labels_real, batch_size=256, verbose=False)
        
        return d_loss, d_accuracy, g_loss
    
    '''
    以下2つのsave_me()とchange_me()は、モデルの保存と（他の訓練済モデルインスタンスからの）移植。
    DCGANの場合、単純にkerasのmodel.save()とload_model()をdiscriminatorとgeneratorでそれぞれやればいい、とはならない。
    本質的には以下の理由である。
    ①「Model」に相当するものは、discriminatorとgeneratorと両者を含むcombinedモデルの3つあるが、
    　dcgan_fmnist_krインスタンス生成時、discriminatorのtrainable設定と各Modelのcompileには、以下のように守るべき順番がある。
     （__init()__参照）
     　discriminatorとgeneratorのインスタンス生成→discriminatorのみcompile→discriminator.trainable=Falseに→
      　→combinedモデルのインスタンスを、discriminatorとgeneratorを含む形で生成→combinedモデルをcompile
    ②kerasのload_model()は、loadしたモデルを問答無用に自動でcompileする（loadしたモデルがcompile済なら）。
    上記②はどうしようもないので、移植時に擬似的に①が達成できるように、保存と移植のそれぞれで工夫する。
    '''
    
    def save_me(self, files_dir, g_file_name="", d_file_name="", val_file_name=""):
        #discrimonatorとgeneratorの両モデル、及びこのDCGAN_FMNIST_krインスタンス内の変数のファイル保存
        #files_dir：これらのファイルの場所。「/」で終わるように。
        #hoge_file_name：discrimonator、generator、その他の変数を格納するファイルの名前。拡張子も含む。        
        
        #ファイル名の指定が無い場合は、このDCGAN_FMNIST_krインスタンスのnameを使用。
        if g_file_name=="":
            g_file_name = self._name + "_g.h5"                    
        if d_file_name=="":
            d_file_name = self._name + "_d.h5"                    
        if val_file_name=="":
            val_file_name = self._name + "_val.pickle"            
            
        g_file_path = files_dir + g_file_name
        d_file_path = files_dir + d_file_name 
        val_file_path = files_dir + val_file_name
        
        #一旦、discriminatorをtrainable=Trueにする。
        #discriminatorをload_model()した時に、kerasは自動でcompileする。
        #その時に、discriminator.trainable=Falseだと、discriminator自体のtrainがなされない。
        #＜discriminator保存時＞discriminator.trainable=True→discriminatorを保存→discriminator.False　
        #Falseに戻さなくても、discriminatorも_gan_combinedもcompile済なので動作はするが、外見と実体の不一致は良くない。
        #ここでFalseに戻しても再compileしなくてよい。Falseに戻しただけだから。
        #＜discriminatorのload時＞discriminatorをload（trainable=Trueになっている）→kerasがdiscriminatorをcompile→
        #discriminator.trainable=Falseにする→コンテナの_gan_combinedを再構築（新規インスタンス化）→_gan_combinedをcompile
        
        #generatorの保存
        self._generator.save(g_file_path)
        
        #discriminatorの保存
        try:
            #一旦、discriminatorをtrainable=Trueに
            self._discriminator.trainable = True
            #discriminatorのsave
            self._discriminator.save(d_file_path)
        finally:
            #discriminatorをtrainable=Falseに戻す
            self._discriminator.trainable = False
                    
        #参考
        #https://note.nkmk.me/python-tensorflow-keras-trainable-freeze-unfreeze/
        #インスタンス化後にレイヤーのtrainableプロパティにTrueかFalseを設定することができます．
        #設定の有効化のためには，trainableプロパティの変更後のモデルでcompile()を呼ぶ必要があります．
        
        #valの保存
        
        #valのdictionary生成
        val = {}
        val["name"] = self._name
        val["img_shape"] = self._img_shape
        val["z_dim"] = self._z_dim
        val["z_fixed_for_sample"] = self._z_fixed_for_sample        
        #valのsave
        save_pickle_file(val, val_file_path)
        
    def change_me(self, files_dir, g_file_name, d_file_name, val_file_name):
        #discrimonatorとgenerator、及びdcgan_fmnist_krインスタンス変数の保存済ファイルを読み込み、このモデル内で継続使用可能にする。
        
        g_file_path = files_dir + g_file_name
        d_file_path = files_dir + d_file_name
        val_file_path = files_dir + val_file_name
        
        #dcgan_fmnist_krインスタンス変数のファイル読み込み
        val = read_pickle_file(val_file_path)
        #val各変数を一旦tempに入れる
        temp_name = val["name"] #名前
        temp_img_shape = val["img_shape"] #画像のshape
        temp_z_dim = val["z_dim"] #z_dim
        temp_z_fixed_for_sample = val["z_fixed_for_sample"] #サンプルイメージ生成用の固定潜在変数z
        
        #generatorインスタンス復元
        temp_generator = load_model(g_file_path)
        #保存時、generator自体はcompileされていないはず。なので、kerasもtemp_generatorをcompileしないはず。
        
        #discriminatorインスタンス復元
        temp_discriminator = load_model(d_file_path)
        #保存時、generator自体はcompile済なので、kerasはtemp_discriminatorを自動でcompileする。
        #保存時に一瞬だけdiscriminator.trainable = Trueにしたので、temp_discriminator.trainable=Trueのはず。
        #このtemp_discriminator.trainable=Trueの状態でkerasがtemp_discriminatorを自動compileするようにする。
        
        #以下、__init()__と大体同じことをする。
        
        #temp_discriminatorを訓練対象外に
        temp_discriminator.trainable = False
        
        #コンテナの_gan_combinedを再構築。新規にインスタンスを生成する。
        temp_gan_combined = self._prepare_combined_models_gan(temp_generator, temp_discriminator)
        #_gan_combinedを使用可能に
        temp_gan_combined.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)) 
        
        self._generator = temp_generator
        self._discriminator = temp_discriminator    
        self._gan_combined = temp_gan_combined
        
        self._name = temp_name
        self._img_shape = temp_img_shape
        self._z_dim = temp_z_dim
        self._z_fixed_for_sample = temp_z_fixed_for_sample
        
    def _summarize_models(self, g=True, d=True, comb_gan=True):
        #内部の3つのモデルのsummaryを出力
        if g==True:
            print("Generator")
            self._generator.summary()
        if d==True:
            print("Discriminator")
            self._discriminator.summary()
        if comb_gan==True:
            print("Combined GAN")
            self._gan_combined.summary()            
    
    def generate_z(self, len_z):
        #潜在変数zを生成して返す。モデルの利用者の利便性のため。
        z = np.random.normal(0, 1, (len_z, self._z_dim))
        return z
    
    def train(self, imgs_train, epochs, batch_size=128, ls_epsilon=0, sample_img_interval=0):
        #訓練関数
        #imgs_train：訓練データである画像データ。shapeは(画像枚数, *self._img_shape)。値域は[0, 1]か「0, 255」であること。
        #sample_img_interval：何エポック毎にgeneratorによるサンプルイメージ出力と表示を行うか。0以下だと行わない。
        
        start_time = datetime.now()
        
        #画像データのスケールを調整する。値域を(-1, 1)にする。
        imgs_train = convert_scale_img_data(imgs_train)
        
        num_imgs_train = imgs_train.shape[0]
        iters = np.ceil(num_imgs_train / batch_size).astype(np.int) #イテレーション数
        
        #以下は、記録用        
        processing_time_epochs = [] #1エポックに費やした時間（秒）　履歴
        d_loss_epochs = [] #各エポックでのdiscriminatorのloss　履歴
        d_accuracy_epochs = [] #各エポックでのdiscriminatorのaccuracy　履歴
        g_loss_epochs = [] #各エポックでのgeneratorのloss　履歴
        probs_fake_epochs = [] #各エポック終了時、discriminatorが偽物画像を本物であるとした確率　履歴
        probs_real_epochs = [] #各エポック終了時、discriminatorが本物画像を本物であるとした確率　履歴

        ###各エポック後のエポック成果計測の準備###

        #各エポックの終わりに、そのエポックの成果として、lossとaccuracyを算出する。
        #そのためには、固定のzと固定の本物画像が必要である。
        #固定の本物画像については、全訓練データを使用するのが理想であるが、重いデータを毎エポックで順伝播させると動作も重くなる。
        #よって、固定の本物画像は、全訓練データの画像枚数の一部とする。固定のzの個数も同じとする。
        num_data_fixed_half = np.ceil(num_imgs_train / (5*2)).astype(np.int) #固定のz個数の半分(=偽物画像の枚数、固定の本物画像の枚数)

        #固定のz個数の半分(=偽物画像の枚数、固定の本物画像の枚数)　の決定
        if num_imgs_train<2500:
            #全訓練データ枚数2500枚未満なら、全部
            num_data_fixed_half = np.floor(num_imgs_train / 2).astype(np.int) 
        elif num_imgs_train<10000:
            #全訓練データ枚数10000枚未満なら、その約4分の1
            num_data_fixed_half = np.floor(num_imgs_train / (4*2)).astype(np.int) 
        else:
            #全訓練データ枚数10000枚以上なら、2500
            num_data_fixed_half = 1250

        z_fixed_for_g = np.random.normal(0, 1, (num_data_fixed_half*2, self._z_dim)) #固定の潜在変数z(g_loss_epoch用) (num_data_fixed_half*2, self._z_dim)
        label_real_fixed_for_g = np.ones((num_data_fixed_half*2,1)) #固定の本物ラベル(g_loss_epoch用) (num_data_fixed_half*2,1)　本物ラベルは「1」で満たす。
        idx_all = np.arange(num_imgs_train)
        np.random.shuffle(idx_all)
        idx_fixed_half = idx_all[0:num_data_fixed_half]
        imgs_real_fixed = imgs_train[idx_fixed_half] #固定の本物画像 (num_data_fixed_half, 28, 28, 1)
        label_fake_fixed = np.zeros((num_data_fixed_half,1)) #固定の偽物ラベル (num_data_fixed_half,1)　偽物ラベルは「0」で満たす。
        label_real_fixed = label_real_fixed_for_g[0:num_data_fixed_half] #固定の本物ラベル (num_data_fixed_half,1)　本物ラベルは「1」で満たす。
        label_fixed = np.concatenate([label_real_fixed, label_fake_fixed], 0) #固定のラベル (num_data_fixed_half*2, 1)
        ###各エポック後のエポック成果計測の準備　終わり###        

        iter_count = 0 #イテレーション総回数
                
        for epoch in range(epochs):  # e:1エポック 
            
            print("\nEpoch:", epoch)
                        
            #エポック開始日時を取得
            epoch_start_dt = datetime.now()
            
            #イテレーション毎のミニバッチ抽出のための、全imgs_trainのインデックスのシャッフル
            idx = np.arange(num_imgs_train)
            np.random.shuffle(idx)
            
            for it in range(iters):
                
                mask = idx[batch_size*it : batch_size*(it+1)]
                batch_size_mb = mask.shape[0] #1エポック内の最後のイテレーションは、指定されたbatch_sizeより少なくなるだろう。
    
                # ミニバッチの生成
                imgs_real_mb = imgs_train[mask]
            
                #本物ラベルと偽物ラベルのベクトルを生成
                #訓練に使用する正解ラベルのみ、本物ラベルは「1」、偽物ラベルは「0」で満たすが、さらに指定されたラベル平滑化のεで加減算する。
                label_real_mb = np.ones((batch_size_mb,1)) - ls_epsilon # ((batch_size_mb,1)
                label_fake_mb = np.zeros((batch_size_mb,1)) + ls_epsilon # ((batch_size_mb,1)　
                
                ###discriminatorの訓練###
                
                #discriminator本体そのものに対して訓練を施す。
                #最初に偽物画像の判定をさせて訓練。
                #潜在変数をbatch_size_mb分生成
                z_mb = np.random.normal(0, 1, (batch_size_mb, self._z_dim)) #(batch_size_mb, self._z_dim)
                #generatorにbatch_size_mb分の偽物画像を生成させる
                imgs_fake_mb = self._generator.predict(z_mb) #(batch_size_mb, 28, 28, 1)
                #batch_size_mb分の偽物画像をdiscriminatorに判定させて訓練。
                d_loss_fake_it, accuracy_fake_it = self._discriminator.train_on_batch(imgs_fake_mb, label_fake_mb)
                #次に本物画像の判定をさせて訓練。
                #batch_size_mb分の本物画像をdiscriminatorに判定させて訓練。
                d_loss_real_it, accuracy_real_it = self._discriminator.train_on_batch(imgs_real_mb, label_real_mb)
                #このイテレーションの結果を算出。lossとaccuracyについて、realとfakeの平均を取る。
                #realとfakeはそれぞれの画像枚数が同じbatch_size_mbであり、realとfakeでMECEなので、平均を取ることには妥当性がある。
                d_loss_it = (d_loss_fake_it + d_loss_real_it) * 0.5
                accuracy_it = (accuracy_fake_it + accuracy_real_it) * 0.5
                                
                ###generatorの訓練###
                
                #generator本体ではなく、__gan_combinedに対して訓練を施す。discriminatorは訓練対象外なので、誤差逆伝播の通り道でしかない。
                #_gan_combinedで一気通貫　<順伝播>潜在変数→generator偽物生成→discriminator判定→<逆伝播>誤差→discriminator→generator
                #潜在変数をbatch_size_mb分生成
                z_mb = np.random.normal(0, 1, (batch_size_mb, self._z_dim)) #(batch_size_mb, self._z_dim)
                #_gan_combinedで一気通貫に順伝播→一気通貫に誤差逆伝播＆generatorのみ訓練（パラメーター更新）
                g_loss_it = self._gan_combined.train_on_batch(z_mb, label_real_mb) #偽物ではなく本物ラベル「1」　Non-saturating GAN
                #↑generatorの損失関数を変更　「Non-saturating GAN」
                #従来の損失関数　Σlog( 1-D(G(z)) ) 　⇒　-Σlog(D(G(z)))　に変更。「Non-saturating GAN」と呼ばれる。
                #従来の損失関数だとgeneratorの訓練が進まない（特に初期）。
                #-Σlog(D(G(z)))　は、出力全件に対する正解ラベルが全部「1」のみの場合のBinaryCrossEntropyLoss値にもなっている。
                #よって、上記のg_loss_itの実装となる。ただ、この実装は既に世間一般で広く行われているようである。
                
                iter_count += 1
                
            #イテレーションのfor　終わり
            
            ###各エポック後のエポック成果計測###
            
            #エポックの成果として、lossとaccuracyを計測する。d_loss, d_accuracy, g_loss。
            #あらかじめ準備してあった、固定のz、固定の本物画像、固定のラベルを使用する。
            #ただし偽物画像については、この訓練状態で生成したものでなければ意味は無いので、ここで生成。
            
            #偽物画像生成
            z_this_epoch_for_fake = np.random.normal(0, 1, (num_data_fixed_half, self._z_dim))
            imgs_fake_this_epoch = self._generator.predict(z_this_epoch_for_fake) #固定の偽物画像 (num_data_fixed_half, 28, 28, 1)
            
            #固定の本物画像と今生成した偽物画像の連結 
            imgs_real_fixed_fake_this_epoch = np.concatenate([imgs_real_fixed, imgs_fake_this_epoch], 0) #(num_data_fixed_half*2, 28, 28, 1)
                        
            #discriminatorのlossとaccuracyの計測
            d_score_epoch = self._discriminator.evaluate(x=imgs_real_fixed_fake_this_epoch, y=label_fixed, batch_size=256, verbose=False)
            d_loss_epoch, d_accuracy_epoch = d_score_epoch[0], d_score_epoch[1] 
            #generatorのlossの計測　偽物でなく本物ラベルを使用することに注意。理由はg_loss計算のコメント参照。
            g_loss_epoch = self._gan_combined.evaluate(x=z_fixed_for_g, y=label_real_fixed_for_g, batch_size=256, verbose=False)

            #discriminatorがここで生成された偽物画像を本物と推定した確率（平均）
            probs_fake_epoch = np.mean(self.discriminate(imgs_fake_this_epoch))#[0:3]))
            #discriminatorが特定の本物画像を本物と推定した確率（平均）
            probs_real_epoch = np.mean(self.discriminate(imgs_real_fixed))#[0:3]))
            
            ###各エポック後のエポック成果計測　終了###
                        
            d_loss_epochs.append(d_loss_epoch)
            d_accuracy_epochs.append(d_accuracy_epoch)
            g_loss_epochs.append(g_loss_epoch)
            probs_real_epochs.append(probs_real_epoch)
            probs_fake_epochs.append(probs_fake_epoch)
            
            print(" iteration count:", iter_count)
            print(" g_loss_epoch:", g_loss_epoch)
            print(" d_loss_epoch:", d_loss_epoch) 
            print(" d_accuracy_epoch:", d_accuracy_epoch) 
            print(" Discriminatorは偽物画像" + str(imgs_fake_this_epoch.shape[0]) + "枚を平均" + str(probs_fake_epoch) + "の確率で本物と推定")
            print(" Discriminatorは本物画像" + str(imgs_real_fixed.shape[0]) + "枚を平均" + str(probs_real_epoch) + "の確率で本物と推定")
                        
            #指定されたintervalで、サンプルイメージを生成
            if sample_img_interval>0 and ( epoch%sample_img_interval==0 or (epoch+1)==epochs ):
                #最初のエポック、以降は指定されたintervalのエポック毎、最終エポックで、サンプルイメージを生成し表示
                #サンプルイメージ生成　必ず以下のpublic関数を通す（出力データ値域を画像データとして適正な[0,1]に変換している）
                imgs_sample = self.generate(self._z_fixed_for_sample)
                #サンプルイメージを表示
                show_images(imgs_sample, img_shape=self._img_shape, rows=4, cols=8, channels_first=False)
            
            epoch_end_dt = datetime.now()
            processing_time_epoch = epoch_end_dt - epoch_start_dt
            processing_time_epochs.append(processing_time_epoch)
            print(" Processing time(seconds):", processing_time_epoch.seconds)
            print(" Time:", datetime.now().strftime('%H:%M:%S'))            
            
        #エポックのfor　終わり
        
        end_time = datetime.now()        
        
        processing_time_total = end_time - start_time
        processing_time_total_string = timedelta_HMS_string(processing_time_total)
        print("\nTotal Processing time:", processing_time_total_string)
        
        #結果をまとめたオブジェクト「result」を返す
        result = {}
        result["name"] = self._name
        result["epochs"] = epochs
        result["batch_size"] = batch_size
        result["ls_epsilon"] = ls_epsilon
        result["d_loss_epochs"] = d_loss_epochs
        result["d_accuracy_epochs"] = d_accuracy_epochs        
        result["g_loss_epochs"] = g_loss_epochs
        result["probs_real_epochs"] = probs_real_epochs
        result["probs_fake_epochs"] = probs_fake_epochs
        result["processing_time_epochs"] = processing_time_epochs
        result["processing_time_total"] = processing_time_total_string #時分秒の文字列が入るので注意。
        result["processing_time_total_td"] = processing_time_total
        result["iter_count"] = iter_count

        return result
    
    @property
    def img_shape(self):        
        return self._img_shape
    
    @property
    def z_dim(self):        
        return self._z_dim
    
    @property
    def name(self):        
        return self._name
    @name.setter
    def name(self, name):
        self._name = name
    
    @property
    def z_fixed_for_sample(self):
        #訓練時のサンプルイメージ生成用の潜在変数z
        return self._z_fixed_for_sample