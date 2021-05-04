# DCGAN Fashion-MNISTの模倣画像を生成 （tensorflow.kerasを使用）

tensorflow.kerasで構築したDCGANです。<br>
Fashion-MNISTの模倣画像を生成します。<br><br>
![real_vs_fake3](https://user-images.githubusercontent.com/52105933/91722778-b5dde100-ebd5-11ea-9398-b29ef9094590.png)

<BR>
  
## 概要
tensorflow.kerasで構築したDCGANです。<br>
Fashion-MNISTの模倣画像を生成します。<br>
訓練　→　画像生成　→　モデル保存　→　再利用　という一連のサイクルに対応した機能をpublicインターフェース上にはじめから備えており、これらの機能を作り込むこと無く手軽に使用することができます。<br>
DiscriminatorとGeneratorのlayer構成を変えるだけで、他の画像データセット向けのDCGANとして訓練・利用することができます。<BR>

※理論の説明は基本的にしていません。他のリソースを参考にしてください。<br>
&nbsp;&nbsp;ネットや書籍でなかなか明示されておらず、私自身が実装に際し情報収集や理解に不便を感じたものを中心に記載しています。

<BR>
  
## 訓練の成果

#### 生成画像
![fake](https://user-images.githubusercontent.com/52105933/93406729-5ff28400-f8cb-11ea-8fcb-c6da7955b385.png)

#### GeneratorとDiscriminatorのLoss
![training_20200827_03_loss](https://user-images.githubusercontent.com/52105933/93406849-b233a500-f8cb-11ea-979c-977abd6f2bce.png)

#### DiscriminatorのAccuracy
![training_20200827_03_accuracy](https://user-images.githubusercontent.com/52105933/93406903-d8594500-f8cb-11ea-8f9b-b55a5f015908.png)

#### Discriminatorは本物画像（「real」）と生成画像（「fake」）それぞれをどのくらいの確率で本物と推定したか
![training_20200827_03_probability](https://user-images.githubusercontent.com/52105933/93407060-2f5f1a00-f8cc-11ea-88da-dc88ea4169bc.png)

<BR>

## 訓練の内容

### DiscriminatorとGeneratorの訓練

以下の通りです。
| 訓練したい方 | 実際に訓練を施す方 |
| :---         | :---         | 
|Discriminator|Discriminator|
|Generator|コンテナのCombined Model<br>（GeneratorとDiscriminatorを一気通貫で順伝播＆誤差逆伝播するから）|

<br>

- Discriminator訓練時<BR>
Combined ModelにではなくDiscriminator自身に訓練を施します。<br>（誤差逆伝播はDiscriminator自身で止まるため）<br>
Discriminator.trainable=Falseですが、Discriminator.trainable=Trueの時にcompileしてあるので、単体では訓練されます。<br>
<br>![train_disc](https://user-images.githubusercontent.com/52105933/91540602-fdeed080-e955-11ea-9d2f-da803e49b321.png)

<BR>
  
- Generator訓練時<BR>
GeneratorにではなくCombined Modelに訓練を施します。<br>（Generator→Discriminatorと順伝播し、帰り道はDiscriminator→Generatorと一気通貫に誤差逆伝播するから）<br>
Discriminator.trainable=Falseの状態でCombined Modelをcompileしているので、Discriminatorは誤差逆伝播の通り道になるだけで訓練はされず（パラメーターは更新されず）、Generatorのみ訓練されます。<br>
<br>![train_gen](https://user-images.githubusercontent.com/52105933/91557082-ca6c7000-e96e-11ea-93ac-00eef1219e5c.png)

<br>

### Generatorの損失関数の変更とその実装　「Non-Saturating GAN」
同一の評価関数をプラスとマイナスと符号だけ逆転させて、DiscriminatorとGeneratorで綱引し合う従来のGANは「Min-Max GAN」と呼ばれます。<br>
  
- **Min-Max GANのGeneratorの損失関数：Σ( log( 1 – D(G(z_i)) ) )** <br>
この損失関数は、（特に訓練初期は）Generatorの訓練がほとんど進まないことが知られています。 <br>
訓練初期は、D(G(z))がとても低いからです（Discriminatorは訓練初期でまだ未熟とはいえ、同じく未熟なGeneratorが生成した砂嵐画像を本物判定しないため）。

そこで、以下のGeneratorの損失関数が考え出されました。これが「Non-Saturating GAN」です。<br>

- **Non-Saturating GANのGeneratorの損失関数：-Σ( log( D(G(z_i)) ) )** <br>
訓練初期でD(G(z))がとても低いと、むしろ損失値-log( D(G(z)) )は大きな正の数となり、Generatorの訓練は一気に進みます。<br>

実装について。<br>
Non-Saturating GANのGeneratorの損失関数の-Σ( log( D(G(z_i)) ) )は、「サンプルデータi全件に対応する正解ラベルが全部1（True）のみ」の場合のBinaryCrossEntropyLoss値でもあります。<br>
よって、前掲のごとく、全偽物画像に対して正解ラベル1（本物）を充当する、という実装となります。<br>
※このNon-Saturating GANのGeneratorの損失関数の実装は、既に一般的に広く行われているようです。<BR>

まとめると、以下の通りです。
| GANの種類 | Generatorの損失関数 |実装概要|
|   :---:   | :---:  |  :---          |
|Min-Max GAN (従来の）|Σ( log( 1 – D(G(z_i)) ) )|（むしろこちらの実装を見たことがありません）|
|Non-Saturating GAN|-Σ( log( D(G(z_i)) ) )|全偽物画像に対して正解ラベル1（本物）を充当|

<BR>
  
### 正解ラベルのラベル平滑化
訓練に使用する正解ラベルは、本物なら「1」、偽物なら「0」の数値になっています。<BR>
これを、「ラベル平滑化（label smoothing）」すると、モデルの性能が良くなった、という報告が一部にあるので、実装してみました。<br><BR>
DCGANにおけるラベル平滑化は、以下の通りです。<br>
ラベル平滑化のε=0.1として（これはtrain()の引数「ls_epsilon」に相当）、<br>
本物の正解ラベル [1, 1, 1, 1] ⇒ε=0.1で平滑化⇒ [0.9, 0.9, 0.9, 0.9]<br>
偽物の正解ラベル [0, 0, 0, 0] ⇒ε=0.1で平滑化⇒ [0.1, 0.1, 0.1, 0.1]<br><BR>
本モデルにも適用し、Fashion-MNISTに使用してみましたが・・・目立った効果はありませんでした。（Fashion-MNIST程度では効果が無い？）

<br>

## 実行確認環境と実行の方法

### 実行確認環境

以下の環境での実行を確認しました。<br>

- Keras 2.3.1
- tensorflow 1.15.0
- h5py 2.10.0

これらのインストールは、各種リソースを参考にしてください。<br>

### 実行の方法

訓練済モデルの使用、訓練と推論の具体的な方法は、ファイル「DCGAN_FMNIST_Keras_demo.ipynb」を参照してください。<br>

<BR>

## ディレクトリ構成
dcgan_fmnist_kr.py<BR>
common/<br>
&nbsp;└tools.py<br>
-----以下デモ用ノートブック関連-----<br>
DCGAN_FMNIST_Keras_demo.ipynb<BR>
demo_model_files/<br>
&nbsp;└（h5ファイルやpickleファイル）<br>
---------------------------------------------------------------<br>
- dcgan_fmnist_kr.py：モデル本体。中身はclass dcgan_fmnist_kr です。モデルを動かすにはcommonフォルダが必要です。
- DCGAN_FMNIST_Keras_demo.ipynb：デモ用のノートブックです。概要をつかむことができます。このノートブックを動かすにはdemo_model_filesフォルダが必要です。
- Fashion-MNISTデータセットは含まれていません。Keras経由でダウンロードすることができます。
  
<BR>
  
## モデルの構成
dcgan_fmnist_kr.pyのclass dcgan_fmnist_kr が、モデルの実体です。<br><br>
![internal_structure](https://user-images.githubusercontent.com/52105933/93408685-1bb5b280-f8d0-11ea-93f6-d17b04d854d6.png)

このclass dcgan_fmnist_kr をアプリケーション内でインスタンス化して、訓練や画像生成といったpublicインターフェースを呼び出す、という使い方をします。<br>
Generator、Discriminator、Combined Modelはdcgan_fmnist_kr内部に隠蔽され、外部から利用することはできません。<br>
Generator、Discriminatorそれぞれのlayer構成を変えるだけで、他の画像データセット向けのDCGANとして訓練・利用することができます。
```
#モデルのインポート 
from dcgan_fmnist_kr import dcgan_fmnist_kr #モデル本体
  
#モデルのインスタンスを生成 
gan_model_instance = dcgan_fmnist_kr(hoge, hoge, …)

#以下、モデルインスタンスに対してそのpublic関数を呼ぶ

#このモデルインスタンスの訓練 
result = gan_model_instance.train(hoge, hoge, …)

#この訓練済モデルインスタンスから画像生成 
img_gens = gan_model_instance.generate(hoge, hoge, …)

#この訓練済モデルインスタンスの保存
gan_model_instance.save_me(hoge, hoge, …)

#別の訓練済モデルをこのモデルインスタンスに移植
gan_model_instance.change_me(hoge, hoge, …)
```

<br>

### class dcgan_fmnist_kr　のpublicインターフェース一覧
| 名前 | 関数/メソッド/プロパティ | 機能概要・使い方 |
| :---         |     :---:      | :---         |
|dcgan_fmnist_kr|     -      |class dcgan_fmnist_kr　のモデルインスタンスを生成する。<br>*model_instance* = dcgan_fmnist_kr(name="hoge", z_dim=100, img_shape=(28, 28, 1)|
|train|     関数      |モデルインスタンスを訓練する。<br>result = *model_instance*.train(imgs_train=img_train_array, epochs=40, batch_size=128, ls_epsilon=0, sample_img_interval=0)|
|generate|     関数      |モデルインスタンスが画像を生成する。<br>img_generated = *model_instance*.generate(z=z)|
|save_me|     メソッド      |モデルインスタンスをファイル保存する。<br>*model_instance*.save_me(files_dir="hoge_dir/", g_file_name="hoge_g", d_file_name="hoge_d", val_file_name="hoge_val")|
|change_me|     メソッド      |モデルインスタンスに、別のモデルインスタンスを取り込む（移植する）。<br>*model_instance*.change_me(files_dir="hoge_dir/", g_file_name="hoge_g", d_file_name="hoge_d", val_file_name="hoge_val")|
|discriminate|     関数      |モデルインスタンスが、与えられた画像が本物である確率を判定する。モデルインスタンスの性能計測用。<br>probs = *model_instance*.discriminate(imgs=imgs_array)|
|evaluate|     関数      |モデルインスタンスが、与えられた画像と正解ラベルから、DiscriminatorのLossとAccuracy、GeneratorのLossを算出する。モデルインスタンスの性能計測用。<br>d_loss, d_accuracy, g_loss = *model_instance*.evaluate(imgs=imgs_array, labels=labels_array)|
|generate_z|     関数      |モデルインスタンスが、画像生成に使われる潜在変数を生成する。モデルユーザーのための便利機能。<br>z = *model_instance*.generate_z(len_z=20)|
|img_shape|     getterプロパティ      |モデルインスタンスが認識している、対象とする画像のshape。インスタンス化時に指定された物。<br>img_shape = *model_instance*.img_shape|
|z_dim|     getterプロパティ      |モデルインスタンスが認識している、材料とする潜在変数の次元。インスタンス化時に指定された物。<br>z_dim = *model_instance*.z_dim|
|name|     getter/setterプロパティ      |モデルインスタンスの名前。<br>getter : hoge = *model_instance*.name<br>setter : *model_instance*.name = hoge|
|z_fixed_for_sample|     getterプロパティ      |モデルインスタンスが訓練時にサンプルイメージを生成する際に使用している、固定の潜在変数。<br>z_fixed = *model_instance*.z_fixed_for_sample|

<br><br>
※本リポジトリに公開しているプログラムやデータ、リンク先の情報の利用によって生じたいかなる損害の責任も負いません。これらの利用は、利用者の責任において行ってください。