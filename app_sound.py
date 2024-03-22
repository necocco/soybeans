import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import Net
import time
from PIL import Image, ExifTags
import base64
import pygame
import random
import os
from PIL import Image
from torchvision import transforms
from glob import glob
from natsort import natsorted
from torchvision.models import resnet34
from torchmetrics.functional import accuracy
import torch
import streamlit as st
import pandas as pd
import plotly.graph_objects as govenv
import plotly.express as px

# モデルの定義
# ここでモデルのアーキテクチャを定義して、重みをロードする必要があります
# クラスラベルの定義（絵文字も含む）
labels = {
    0: "Really?　Soybeans?? 👻　なんだキミは？",
    1: "わるい子　😭",
    2: "いい子　😍",
    3: "ふつうの子　😑"
}


# Streamlitアプリケーションの定義
# CSSを使って背景色を設定する関数
def set_bg_color():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color:#FFFFCC;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 関数を呼び出して背景色を設定
set_bg_color()

import streamlit as st

st.sidebar.write(':blue[**大豆選別のための画像判別アプリ**]')
st.sidebar.write("予測結果は以下の４つに分けられます")
st.sidebar.write("**・いい子**：良質な大豆")
st.sidebar.write("**・ふつうの子**：味噌等の加工用に使える大豆")
st.sidebar.write("**・わるい子**：品質が悪い大豆。鳥にあげましょう")
st.sidebar.write("**・Really?　Soybeans??**：あなたはだぁれ？")

# 画像の読み込み
img = Image.open("./IMG_9183-1.JPG")

# Exif情報を取得し、向き情報を確認する
try:
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            exif = dict(img._getexif().items())

            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
except (AttributeError, KeyError, IndexError):
    # Exif情報がない場合や、向き情報が取得できない場合は何もしない
    pass

# 画像を表示する
st.sidebar.image(img, use_column_width=True)

def main():
    # Pygameの初期化
    pygame.init()
    pygame.mixer.init()


    # コンテンツを追加
    st.markdown("<div class='content-container'>", unsafe_allow_html=True)
    st.subheader('大豆画像判定～ よい子・わるい子・ふつうの子 ', divider='rainbow')
    st.title('_:red[Soybeans checker] app_ :mag:')
    st.markdown("</div>", unsafe_allow_html=True)

    

    # 画像ファイルが保存されているディレクトリ
    image_directory = "./images/input/test"

    # 画像ファイルの一覧を取得
    image_files = os.listdir(image_directory)

    # 画像ファイルの一覧を表示してユーザーにダウンロードするように指示
    selected_image_file = st.selectbox("①画像を選択してください:", image_files)

    # ユーザーが選択した画像ファイルのパスを構築
    image_path = os.path.join(image_directory, selected_image_file)

    # 選択された画像ファイルを読み込む
    image = Image.open(image_path)


    st.write('②startボタンをクリックしてください')
    button = st.button('start！')

    if button:
     
        # 音声ファイルのリストを準備する
        audio_files = ['./sound/resnet34afrobeat.mp3','./sound/resnet34groovyRumba1mer.mp3','./sound/ResNet34romanticFolk.mp3', './sound/resnet34chillReggae.mp3','./sound/resnet34sanba1.mp3','./sound/resnet34sanba2.mp3','./sound/ResNet34Folk.mp3','./sound/ResNet34upliftingMetal.mp3']
        
        # ランダムに音声ファイルを選択する
        audio_path = random.choice(audio_files) 
        audio_placeholder = st.empty()

        file_ = open(audio_path, "rb")
        contents = file_.read()
        file_.close()

        audio_str = "data:audio/ogg;base64,%s" % (base64.b64encode(contents).decode())
        audio_html = """
                        <audio autoplay=True>
                        <source src="%s" type="audio/ogg" autoplay=True>
                        Your browser does not support the audio element.
                        </audio>
                    """ % audio_str

        audio_placeholder.empty()
        time.sleep(0.5) #これがないと上手く再生されません
        audio_placeholder.markdown(audio_html, unsafe_allow_html=True)



        # 画像の回転情報を無視して開く
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(image._getexif().items())

            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            # 画像にExif情報がない場合や、回転情報がない場合はスキップ
            pass
        
        #st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.image(image, caption='Uploaded Image.', width=250)

        
        
        # 画像の前処理
        def preprocess_image(image):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # モデルに合わせて画像のサイズを調整
                transforms.ToTensor(),  # 画像をテンソルに変換
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 画像を正規化
            ])
            image = transform(image).unsqueeze(0)  # バッチの次元を追加
            return image


          #予測までのバーを挿入
        progress_text = "🎵なにがでるかな　なにがでるかな　それはResNet34任せよ～"
        my_bar = st.progress(0, text=progress_text) 

        st.write('© Made With Suno')
        
        for percent_complete in range(100):
            time.sleep(0.13)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()

        st.button("Rerun")

        if image is not None:
            # 画像の前処理
            image_tensor = preprocess_image(image)
            # ネットワークの準備
            net = Net().cpu().eval() 
            # 学習済みモデルの読み込み
            net.load_state_dict(torch.load('mnist-rn34-2.pt', map_location=torch.device('cpu')))

    

        # 判定
        with torch.no_grad():
            outputs = net(image_tensor)
            print(outputs)
            _, predicted = torch.max(outputs, 1)
          #  print(predicted.item())
            prediction = labels[predicted.item()]
            prediction_class = prediction 
        st.write(f"予測: {prediction}")
        # st.write(f"予測: {outputs}")
 
        # 画像ファイル名を取得
        image_name = os.path.basename(selected_image_file)

        # 画像ファイル名と正解値の対応表
        image_to_label = {
            "01.jpg": "ふつうの子　😑",
            "02.jpg": "Really?　Soybeans?? 👻　なんだキミは？",
            "03.jpg": "Really?　Soybeans?? 👻　なんだキミは？",
            "04.jpg": "ふつうの子　😑",
            "05.jpg": "いい子　😍",
            "06.jpg": "ふつうの子　😑",
            "07.jpg": "Really?　Soybeans?? 👻　なんだキミは？",
            "10.jpg": "ふつうの子　😑",
            "08.jpg": "いい子　😍",
            "09.jpg": "わるい子　😭",
            "11.jpg": "いい子　😍",
            "12.jpg": "Really?　Soybeans?? 👻　なんだキミは？",
            "13.jpg": "ふつうの子　😑",
            "14.jpg": "わるい子　😭",
            "15.jpg": "Really?　Soybeans?? 👻　なんだキミは？",
            "16.jpg": "Really?　Soybeans?? 👻　なんだキミは？",
            "17.jpg": "わるい子　😭",
            "18.jpg": "いい子　😍",
            "19.jpg": "わるい子　😭",
            "20.jpg": "Really?　Soybeans?? 👻　なんだキミは？",
            "21.jpg": "Really?　Soybeans?? 👻　なんだキミは？",
            "22.jpg": "わるい子　😭",
            "23.jpg": "Really?　Soybeans?? 👻　なんだキミは？",
             }

        #画像ファイル名と音源の対応表
        image_to_audio = {
            "01.jpg": "./sound/normalChild.mp3",
            "02.jpg": "./sound/kurumi.mp3",
            "03.jpg": "./sound/ninniku.mp3",
            "04.jpg": "./sound/normalChild.mp3",
            "05.jpg": "./sound/goodChild.mp3",
            "06.jpg": "./sound/normalChild.mp3",
            "07.jpg": "./sound/uzuramame.mp3",
            "08.jpg": "./sound/goodChild.mp3",
            "09.jpg": "./sound/badChild.mp3",
            "10.jpg": "./sound/normalChild.mp3",
            "11.jpg": "./sound/goodChild.mp3",
            "12.jpg": "./sound/kuroninniku.mp3",
            "13.jpg": "./sound/normalChild.mp3",
            "14.jpg": "./sound/badChild.mp3",
            "15.jpg": "./sound/hanamame.mp3",
            "16.jpg": "./sound/kurogoma.mp3",
            "17.jpg": "./sound/badChild.mp3",
            "18.jpg": "./sound/goodChild.mp3",
            "19.jpg": "./sound/badChild.mp3",
            "20.jpg": "./sound/sirohanamame.mp3",
            "21.jpg": "./sound/kuroninniku.mp3",
            "22.jpg": "./sound/badChild.mp3",
            "23.jpg": "./sound/aobatamame.mp3",
            # 他の画像ファイル名とそれに対応する音源ファイル名も追加します
            }
        

        #画像に対応する正解値を取得
        if selected_image_file in image_to_label:
            label = image_to_label[image_name]
            st.write("正解:", label)

         # Streamlitアプリの正解値
            true_value = label  # ここに正解の値を設定

        if  prediction_class == true_value:
            st.write("**予測が正解と一致しました。**")

            # 一致した場合の音を再生
            sound_file_correct = "./sound/levelUP.mp3"  # 一致した場合の音声ファイルのパスを設定
            pygame.mixer.music.load(sound_file_correct)
            pygame.mixer.music.play()

            # 正解音が終わるまで待機
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            # 正解音が終わった後に画像に対応する音声を再生
            if selected_image_file in image_to_audio:
                audio_path = image_to_audio[selected_image_file]
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                st.write('©ナレーション：音読さん')

            # 音声再生が終了するまで待機
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        else:
            st.write("**予測が正解と一致しません。**")

            # 異なった場合の音を再生
            sound_file_incorrect = "./sound/clumsy2.mp3"  # 異なった場合の音声ファイルのパスを設定
            pygame.mixer.music.load(sound_file_incorrect)
            pygame.mixer.music.play()

            # 音声再生が終了するまで待機
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            # 画像に対応する音声を再生
            if selected_image_file in image_to_audio:
                audio_path = image_to_audio[selected_image_file]
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                st.write('ナレーション　音読さん')

            # 音声再生が終了するまで待機
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
    
                
        #予測確率をグラフで表示   
        # 予測されたテンソルの値を取得
        predicted_values = outputs.squeeze().tolist()

        # 予測されたテンソルの値をPandasのDataFrameに変換
        data = {'Class': list(labels.values()), 'Probability': predicted_values}
        df = pd.DataFrame(data)

        # マイナスとプラスの場合で色を変える関数を定義
        def color_condition(probability):
            return 'red' if probability < 0 else 'blue'

        # Plotlyを使用して棒グラフを作成し、幅を調整
        fig = px.bar(df,x='Class', y='Probability', text='Probability')
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside') # テキストのフォーマットを指定
        fig.update_layout(height=550, width=500, title='Predicted Probabilities(予測確率)', yaxis=dict(title='予測した確率'))
        # バーの色を設定
        fig.update_traces(marker=dict(color=df['Probability'].apply(color_condition)))
        
        # StreamlitでPlotlyのグラフを表示
        st.plotly_chart(fig)

    
if __name__ == "__main__":
    main()
