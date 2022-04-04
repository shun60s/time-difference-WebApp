# -*- coding: utf-8 -*-


"""
チャイム音をマイクでステレオ録音したWAVファイルを入力して、L channelとR channelの音の時間差をpythonを使って推定して、
その結果画像を表示するFlaskとajaxを使ったWeb Appの動作実験するもの。

"""

# Check version
# Python 3.6.4, 64bit on Win32 (Windows 10)
# Flask 2.0.3
# Werkzeug 2.0.3

import os
from flask import Flask, request, jsonify
from flask import render_template
from werkzeug.utils import secure_filename
import queue
import time
import uuid
import json

from tde1 import *

# wavのアップロード先のディレクトリ
UPLOAD_FOLDER = './uploads'
# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['wav'])
# 結果画像が保存されるディレクトリ
RESULT_FOLDER = './static/figure'
# エラーが発生したときの画像
ERROR_FIGURE = './static/figure/figure/error.png'
# 空白の画像
BLANK_FIGURE = './static/figure/figure/blank.png'
# 多重実行制御用のキュー
singleQueue = queue.Queue(maxsize=1)

app = Flask(__name__)  #, static_folder=RESULT_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['ERROR_FIGURE'] = ERROR_FIGURE
app.config['BLANK_FIGURE'] = BLANK_FIGURE

# アップロードされる容量を制限する
app.config['MAX_CONTENT_LENGTH'] = 2 * 2 * 44100 * 15


# 時間差を推定するインスタンスを作成する
tde= time_difference_estimation('sample_wav/chime_only.wav', 1, 0,save_dir='static/figure/',SHOW_PLOT=False, ShowOntheWay=False, SHOW_PLOT2=False)


def allwed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# トップページ
@app.route("/")
def index():
    return render_template('index-ajax.html', filename=app.config['BLANK_FIGURE'])

# ファイルを受け取る方法の指定
@app.route('/upload', methods=['POST'])
def upload_and_tde():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'file' not in request.files:
            return jsonify(filename=app.config['ERROR_FIGURE'], t_time=0, rt_code=0)
        # データの取り出し
        file = request.files['file']
        # ファイル名がなかった時の処理
        if file.filename == '':
	        return jsonify(filename=app.config['ERROR_FIGURE'], t_time=0, rt_code=0)
        # ファイルのチェック
        if file and allwed_file(file.filename):
        # 危険な文字を削除（サニタイズ処理）
            filename = secure_filename(file.filename)
            #　ファイル名をユニークなもにする
            title=filename
            filename=str(uuid.uuid1()) + '___'+ filename
            #print ('filename, title', filename, title)
            # ファイルの保存
            file_path= os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save( file_path)

            # 実行しているものを1回分だけに制御する
            singleQueue.put(time.time()) 
            # 時間差を推定する　
            rt_code,t_time= tde.main0( file_path, title=title)
            # 制限の解除
            singleQueue.get()
            singleQueue.task_done()
            # 推定結果画像の場所を返す
            if rt_code:
                rt_codes=1
                filename=os.path.join(app.config['RESULT_FOLDER'],filename)
                filename_png= os.path.join(app.config['RESULT_FOLDER'], os.path.splitext(os.path.basename(filename))[0]+'.png')
                return jsonify(filename=filename_png, t_time=t_time, rt_code=rt_code)
            else:  # tdeの中でエラー発生
                rt_codes=0
                return jsonify(filename=app.config['ERROR_FIGURE'], t_time=0, rt_code=rt_code)
            
        else:
            return jsonify(filename=app.config['ERROR_FIGURE'], t_time=0, rt_code=0)


if __name__ == "__main__":
    # サーバーを起動する
    app.run(threaded=True) 
