# DCGANをKerasでマジメに構築してみました

DCGANをKerasでマジメに構築してみました。画像データセットはFashion-MNISTを使用しています。<BR><BR>
![real_vs_fake3](https://user-images.githubusercontent.com/52105933/91722778-b5dde100-ebd5-11ea-9398-b29ef9094590.png)
  
## 概要
Kerasで構築したDCGANです。画像データセットFashion-MNISTを訓練データとして使っています。<br><br>
モデルはclassとして定義され、.pyファイルに収められています。<br>
何らかのアプリケーションにこの.pyファイルをimportして、このモデルのclassをインスタンス化し、そのインスタンスのpublicインターフェースを用いて、訓練や画像の生成、その訓練済モデルインスタンスの保存を行う、という使い方です。<br><br>
画像データセットは、Fashion-MNIST以外でも、1チャンネルのグレー画像で軽い物なら、マトモに動くはずです。MNISTは確認済です。

## ディレクトリ構成・動かすのに必要な物
dcgan_fmnist_kr.py<BR>
DCGAN_FMNIST_Keras_demo.ipynb<BR>
common/<br>
└tools.py<br>
demo_model_files/<br>
└（h5ファイルやpickleファイル）<br>
-------------<br>
- dcgan_fmnist_kr.py：モデル本体。中身はclass dcgan_fmnist_kr です。モデルを動かすにはcommonフォルダが必要です。
- DCGAN_FMNIST_Keras_demo.ipynb：デモ用のノートブックです。概要をつかむことが出来ます。このノートブックを動かすにはdemo_model_filesフォルダが必要です。
- Fashion-MNISTデータセットは含まれていません。scikit-learn経由でダウンロードすることができます。
  
## モデルの構成
dcgan_fmnist_kr.pyのclass dcgan_fmnist_kr が、モデルの実体です。<br><br>
![internal_structure](https://user-images.githubusercontent.com/52105933/91521676-2c0ee900-e933-11ea-8f19-bfa3b6604139.png)

このclass dcgan_fmnist_kr をアプリケーション内でインスタンス化して、訓練や画像生成といったpublic関数を呼び出す、という使い方をします。<br>
**Generator、Discriminator、Combined Modelはdcgan_fmnist_kr内部に隠蔽され、外部から利用することはできません。**
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
## class dcgan_fmnist_kr　のpublicインターフェース

#### class dcgan_fmnist_kr　のpublicインターフェース一覧
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

### class dcgan_fmnist_kr　のインスタンス化　*model_instance* = dcgan_fmnist_kr(name, z_dim=100, img_shape=(28, 28, 1))
class dcgan_fmnist_krのインスタンスを生成する。<br><br>
・引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|name|文字列|必須|このモデルインスタンスの名前。|
|z_dim|整数|100|潜在変数zの次元。<br>今後、このインスタンスは、潜在変数の次元はここで指定されたものであるという前提で挙動する。変更方法は無い。|
|img_shape|tuple|(28, 28, 1)|画像1枚のshape。**チャンネルのaxisは必ず最後であること（Kerasで言うところのdata_format='channels_last'）**<br>今後、このインスタンスは、画像のshapeはここで指定されたものであるという前提で挙動する。変更方法は無い。|

※Discriminator、Generator、Combined Modelのインスタンス化とcompileについて<br>
\__init()\__内で、上記3つのインスタンス化とcompileを行っています。<br>
が、以下2つのことを達成するため（train()参照）、若干トリッキーな手順を踏んでいます。<br>
- Discriminatorの訓練は、Discriminator自身に対して行う。
- Generatorの訓練は、Combine Modelに対して行うが、その時、Discriminatorは訓練対象外。

・＜参考＞\__init()\__内で行っている手順：<br>
Generatorをインスタンス化<br>
Discriminatorをインスタンス化<br>
↓<br>
Discriminatorをcompile（trainable=Trueの状態）<br>
↓<br>
Discriminator.trainable=False　に設定<br>
（trainable=Trueの状態でcompile済なので、Discriminatorが単体として”死ぬ”ことは無い）<br>
↓<br>
Combined Modelをインスタンス化<br>
Combined Modelをcompile<br>
（Discriminator.trainable=FalseなのでCombined ModelにとってDiscriminatorは訓練対象外、Generatorのみ訓練対象）<br>
Generator単体ではcompileは行わない（不要）

### ＜関数＞result = *model_instance*.train(imgs_train, epochs, batch_size=128, ls_epsilon=0, sample_img_interval=0)
モデルを訓練します。<br><br>
・引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|imgs_train|ndarray <BR>(N, \*インスタンス化時指定のimg_shape)|必須|訓練データ。<br>画素値の範囲は[0, 1]でも[0, 255]でもよい。|
|epochs|整数|必須|訓練のエポック数。|
|batch_size|整数|128|1イテレーションのバッチサイズ。|
|ls_epsilon|浮動小数点数|0|訓練に使用する正解ラベルのラベル平滑化のε。値域は[0, 1.0]。|
|sample_img_interval|整数|0|何エポック毎にサンプル画像を生成し表示するか。<br>0以下の場合、生成しない。<br>1以上の場合、最初のエポックと最終エポックでは必ず生成・表示される|

・戻り値「result」（Dictionary）の内部要素：
| key文字列 | 型 | 意味 |
| :---         |     :---:      | :---         |
|name|文字列|このモデルインスタンス名。|
|epochs|整数|train()呼び出し時に指定されたエポック数。|
|batch_size|整数|train()呼び出し時に指定されたバッチサイズ。|
|ls_epsilon|浮動小数点数|train()呼び出し時に指定された、正解ラベルのラベル平滑化のε。|
|d_loss_epochs|list|各エポック後に測定したDiscriminatorのLossのエポック毎の記録。listの1要素は1エポック。|
|d_acuracy_epochs|list|各エポック後に測定したDiscriminatorのAccuracyのエポック毎の記録。listの1要素は1エポック。|
|g_loss_epochs|list|各エポック後に測定したGeneratorのLossのエポック毎の記録。listの1要素は1エポック。|
|probs_real_epochs|list|各エポック後に測定した下記の指標のエポック毎の記録。listの1要素は1エポック。<br>Discriminatorが本物画像をどのくらいの確率で本物と推定したか。|
|probs_fake_epochs|list|各エポック後に測定した下記の指標のエポック毎の記録。listの1要素は1エポック。<br>DiscriminatorがGeneratorがその場で生成した偽物画像をどのくらいの確率で本物と推定したか。|
|processing_time_epochs|list|各エポック後に測定したそのエポックでの処理時間秒数のエポック毎の記録。listの1要素は1エポック。|
|processing_time_epochs|list|各エポック後に測定したそのエポックでの処理時間秒数のエポック毎の記録。listの1要素は1エポック。|
|processing_time_total|文字列|訓練の総処理時間の文字列表現。例）2 hours 5 minutes 35 seconds|
|processing_time_total_td|timedela|訓練の総処理時間のtimedeltaオブジェクト。|

・各エポックでの出力文字列：<br>
```
Epoch: 34　　                                                       →　エポック番号（0から始まる）
 iteration count: 16415　　                                         →　総イテレーション回数
 g_loss_epoch: 2.5533969402313232　　                               →　GeneratorのLoss　エポック終了時の計測値
 d_loss_epoch: 0.25367704033851624　　                              →　DiscriminatorのLoss　エポック終了時の計測値
 d_accuracy_epoch: 0.8895999789237976　　                           →　DiscriminatorのAccuracy　エポック終了時の計測値
 Discriminatorは偽物画像1250枚を平均0.17655188の確率で本物と推定　　 →　そのままの意味　エポック終了時の計測値
 Discriminatorは本物画像1250枚を平均0.8495131の確率で本物と推定　　  →　そのままの意味　エポック終了時の計測値
 Processing time(seconds): 30　　                                   →　このエポックの処理時間（秒）
 Time: 02:00:03　　                                                 →　今の時刻
```

#### 訓練関数の構成
![train_func_structure](https://user-images.githubusercontent.com/52105933/91554115-7448fe00-e969-11ea-90c6-8cac20ebf5fe.png)

#### DiscriminatorとGeneratorの訓練
Discriminatorの訓練については、Discriminator自身に訓練を施します。<br>
Generatorの訓練については、Generator自身にではなくコンテナのCombined Modelに対して訓練を施します。<br>
（GeneratorとDiscriminatorを一気通貫で順伝播＆誤差逆伝播するから）<br><br>
・Discriminator訓練時：<br>
誤差逆伝播はDiscriminator自身で止まります。よって、Combined ModelにではなくDiscriminator自身に訓練を施します。<br>
Discriminator.trainable=Falseですが、Discriminator.trainable=Trueの時にcompileしてあるので、単体では訓練されます。<br>
<br>![train_disc](https://user-images.githubusercontent.com/52105933/91540602-fdeed080-e955-11ea-9d2f-da803e49b321.png)

・Generator訓練時：<br>
Generator→Discriminatorと順伝播し、帰り道はDiscriminator→Generatorと一気通貫に誤差逆伝播します。よって、GeneratorにではなくCombined Modelに訓練を施します。<br>
Discriminator.trainable=Falseの状態でCombined Modelをcompileしているので、Discriminatorは誤差逆伝播の通り道になるだけで訓練はされず（パラメーターは更新されず）、Generatorのみ訓練されます。<br>
<br>![train_gen](https://user-images.githubusercontent.com/52105933/91557082-ca6c7000-e96e-11ea-93ac-00eef1219e5c.png)

#### 各エポック終了時の指標測定と、そのための固定データ準備
各エポック終了時点で、モデルの訓練度合いを見るため、以下の指標を測定します。<br>
エポック間で比較できるように、指標算出の元になるデータは固定とし、エポック反復に入る前に準備します。<br>
- GeneratorのLoss
- DiscriminatorのLossとAccuracy
- DiscriminatorはGeneratorがその場で生成した偽物画像をどのくらいの確率で本物と判定したか
- Discriminatorは本物画像をどのくらいの確率で本物と判定したか

この各エポックでの測定指標がlist化され、訓練関数の戻り値resultの要素（前掲）となります。<br>
※「固定データ」とは何で、量はどのくらいか、は、ソースコードをご覧下さい（あらかじめ取り分けておいた訓練データの画像群の一部や、Generatorに偽物画像をその場で生成させるための固定の潜在変数など）。

#### サンプル画像の生成と表示
指定されたエポック数（train()の引数「sample_img_interval」）毎に、32枚の画像をGeneratorが生成し、表示します。<br><br>
時系列での比較が容易になるように、固定の（毎回同じ）潜在変数を使用して、画像を生成します。<br>
また、最初のエポック（エポック番号0）と最終エポックには、サンプル画像を生成・表示します。従って、<br>
エポック番号が、0 → 0 + sample_img_interval → 0 + sample_img_interval + sample_img_interval → ・・・→最終エポック <br>
の時、サンプル画像を生成・表示します。<br>
ただし、sample_img_interval<=0の場合、サンプル画像は（最初の・最終エポック含め）一切生成しません。

#### Generatorの損失関数の変更とその実装　「Non-Saturating GAN」
同一の評価関数をプラスとマイナスと符号だけ逆転させて、DiscriminatorとGeneratorで綱引し合う古典的なGANは「Min-Max GAN」と呼ばれます。<br><br>
**・Min-Max GANのGeneratorの損失関数：Σlog( 1 – D(G(z_i)) )** <br>
ですが、これだと、（特に訓練初期は）GeneratorのLossがほとんど0のままで、Generatorの訓練がほとんど進みません。 <br>
Discriminatorは、訓練初期でまだ未熟とはいえ、同じく未熟なGeneratorが生成した砂嵐画像を持ってこられても、「これは本物だ」とはなかなか言わないです。<br>
つまり、G(z)≓0で、1 – D(G(z))≓1，即ち損失値log(ほぼ1)≓0、となります。よって、特に訓練初期にGeneratorのLossはほとんど0のままとなってしまいGeneratorの訓練は進まず、引きづられてDiscriminatorの訓練も進まなくなります。<br><br>
そこで、以下のGeneratorの損失関数が考え出されました。これが「Non-Saturating GAN」です。<br>
**・Non-Saturating GANのGeneratorの損失関数：-Σlog( D(G(z_i)) )** <br>
この場合、訓練初期でまだ未熟なDiscriminatorが砂嵐画像を（未熟ではあるものの）容易に「これは偽物だ」と判定すると、D(G(z))≓0、即ち損失値-log( D(G(z)) )は大きな正の数となり、Generatorの訓練は一気に進みます。<br><br>
実装について。<br>
Non-Saturating GANのGeneratorの損失関数の-Σlog( D(G(z_i)) )は、「サンプルデータi全件に対応する正解ラベルが全部1（True）のみ」の場合のBinaryCrossEntropyLoss値でもあります。<br>
よって、前掲のごとく、全偽物画像に対して正解ラベル1（本物）を充当する、という、とてもシンプルですがちょっとわかりにくい実装となります。<br>
※このNon-Saturating GANのGeneratorの損失関数の実装は、既に一般的に広く行われているようです（実装者がそうと意識しているかはわかりませんが）。

#### 正解ラベルのラベル平滑化
訓練に使用する正解ラベルは、本物なら「1」、偽物なら「0」の数値になっています。<BR>
これを、「ラベル平滑化（label smoothing）」すると、モデルの性能が良くなった、という報告が一部にあるので、実装してみました。<br><BR>
DCGANにおけるラベル平滑化は、以下の通りです。<br>
ラベル平滑化のε=0.1として（これはtrain()の引数「ls_epsilon」に相当）、<br>
本物の正解ラベル [1, 1, 1, 1] ⇒ε=0.1で平滑化⇒ [0.9, 0.9, 0.9, 0.9]<br>
偽物の正解ラベル [0, 0, 0, 0] ⇒ε=0.1で平滑化⇒ [0.1, 0.1, 0.1, 0.1]<br><BR>
本モデルにも適用してみましたが・・・目立った効果はありませんでした。（Fashion-MNIST程度では効果が無い？）
<br><br>
長かったですが、これで、訓練関数train()の説明は終わりです。<br>

### ＜関数＞imgs_generated = *model_instance*.generate(z)
与えられた潜在変数zを使用して、モデルインスタンス内部のGeneratorが画像を生成し、返します。<BR><br>
・引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|z|ndarray <BR>(N, インスタンス化時指定のz_dim)|必須|潜在変数。
  
・戻り値：
| 名前 | 型 | 意味 |
| :---         |     :---:     | :---         |
|imgs_generated|ndarray <BR>(N, \*インスタンス化時指定のimg_shape)|生成された画像。<br>「N」即ち引数zのデータ個数（axis=0）と同じ個数の画像が生成される。|

### ＜メソッド＞*model_instance*.save_me(files_dir, g_file_name="", d_file_name="", val_file_name="")
現在のモデルインスタンスの状態をファイル保存し、後に再利用できるようにします。<br>
具体的には、Generatorのファイル、Discriminatorのファイル、その他の変数のファイル、の合計3ファイルを生成、保存します。<br><br>
・引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|files_dir|文字列|必須|3ファイルを保存するディレクトリ。|
|g_file_name|文字列|空文字列|Generatorのファイルのファイル名。<br>空文字列の場合、モデルインスタンス名_g.h5　というファイル名になる。|
|d_file_name|文字列|空文字列|Discriminatorのファイルのファイル名。<br>空文字列の場合、モデルインスタンス名_d.h5　というファイル名になる。|
|val_file_name|文字列|空文字列|変数のファイルのファイル名。<br>空文字列の場合、モデルインスタンス名_val.pickle　というファイル名になる。|

GeneratorとDiscriminatorのファイルはHDF5ファイルです。<br>
変数のファイルはpickleファイルです。モデルインスタンス名、インスタンス化時に指定されたimg_shapeとz_dim、訓練時のサンプル画像生成に使用する固定の潜在変数、が含まれます。

### ＜メソッド＞*model_instance*.change_me(files_dir, g_file_name, d_file_name, val_file_name)
保存された別モデルを、自分自身（モデルインスタンス）に取り込み（「移植」）、再利用できるようにします。<br>
訓練済モデルの再利用などに使用できます。<br>
具体的には、指定されたGeneratorのファイル、Discriminatorのファイル、その他の変数のファイル、の合計3ファイルを取り込んで、これらのファイルのモデルに成ります。<br><br>
・引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|files_dir|文字列|必須|3ファイルが置かれているディレクトリ。|
|g_file_name|文字列|必須|Generatorのファイルのファイル名。|
|d_file_name|文字列|必須|Discriminatorのファイルのファイル名。|
|val_file_name|文字列|必須|変数のファイルのファイル名。|

### ＜関数＞probs = *model_instance*.discriminate(imgs)
与えられた画像が本物である確率を、モデルインスタンス内部のDiscriminatorが判定し、返します。<BR>
このモデルインスタンスの性能計測用です。<BR><br>
・引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|imgs|ndarray <BR>(N, \*インスタンス化時指定のimg_shape)|必須|判定したい画像。<br>画素値の範囲は[0, 1]でも[0, 255]でもよい。|

・戻り値：<BR>
- probs<BR>
与えられた各画像それぞれの、本物である確率。shapeは(N, 1)。

### ＜関数＞d_loss, d_accuracy, g_loss = *model_instance*.evaluate(imgs, labels)
与えられた画像と正解ラベルを使用して、モデルインスタンス内部のDiscriminatorがLossとAccuracyを返します。<BR>
imgsとlabelsと同数の潜在変数を内部で生成して、GeneratorのLossも計算し、返します。<br>
このモデルインスタンスの性能計測用です。<BR><br>
・引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|imgs|ndarray <BR>(N, \*インスタンス化時指定のimg_shape)|必須|DiscriminatorがLossとAccuracyを計算するための画像。<br>画素値の範囲は[0, 1]でも[0, 255]でもよい。|
|labels|ndarray <BR>(N, 1)|必須|引数の画像に対応する正解ラベル。<br>本物が「1」、偽物が「0」。|

・戻り値：<BR>
- d_loss, d_accuracy<BR>
与えられた画像と正解ラベルからDiscriminatorが計算したLossとAccuracy。スカラー。
- g_loss<BR>
与えられた画像と正解ラベルと同数の潜在変数（shapeは(N, インスタンス化時指定のz_dim)）を生成し、Generatorが計算したLoss。スカラー。

### ＜関数＞z = *model_instance*.generate_z(len_z)
指定された個数の潜在変数を生成します。<br>
モデルのユーザーに対する単なる便利機能です。必ずしも使用する必要はありません。<br><br>
・引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|len_z|整数|必須|潜在変数の個数。|

・戻り値：<BR>
- z<br>
平均0、標準偏差1の正規分布に従った、len_z個の潜在変数の配列。shapeは(len_z, インスタンス化時指定のz_dim)。
  
### ＜getterプロパティ＞*model_instance*.img_shape
このモデルインスタンスが認識している、画像のshapeを返します。
インスタンス化時に指定された物です。

### ＜getterプロパティ＞*model_instance*.z_dim
このモデルインスタンスが認識している、潜在変数の次元を返します。
インスタンス化時に指定された物です。

### ＜getter/setterプロパティ＞*model_instance*.name
getterは、このモデルインスタンスの名前を返します。<br>
setterは、このモデルインスタンスの名前を設定します。<br><br>
・setterが受け取る値：
| 型 |  意味 |
|     :---:      | :---         |
|文字列|モデルインスタンスの新しい名前。|

### ＜getterプロパティ＞*model_instance*.z_fixed_for_sample
訓練時にサンプル画像を生成する際、固定の潜在変数が使われます。時系列での比較ができるようにするためです。<br>
このモデルインスタンスが保持しているその固定の潜在変数を返します。<br><br>

※本リポジトリに公開しているプログラムやデータ、リンク先の情報の利用によって生じたいかなる損害の責任も負いません。これらの利用は、利用者の責任において行ってください。