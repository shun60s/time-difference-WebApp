# -*- coding: utf-8 -*-


"""
チャイム音をマイクでステレオ録音したWAVファイルを入力して、L channelとR channelの音の時間差をpythonを使って推定して、
その結果画像を表示するFlaskを使ったWeb Appの動作実験するもの。

"""

# Check version
# Python 3.6.4, 64bit on Win32 (Windows 10)
# Flask 2.0.3
# Werkzeug 2.0.3

import os
from flask import Flask, request, redirect, url_for
from flask import render_template
from flask import send_from_directory
from werkzeug.utils import secure_filename
from flask import send_from_directory
import queue
import time
import uuid

from tde1 import *

# wavのアップロード先のディレクトリ
UPLOAD_FOLDER = './uploads'
# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['wav'])
# 結果画像が保存されるディレクトリ
RESULT_FOLDER = './static/figure'
# 多重実行制御用のキュー
singleQueue = queue.Queue(maxsize=1)

app = Flask(__name__)  #, static_folder=RESULT_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# アップロードされる容量を制限する
app.config['MAX_CONTENT_LENGTH'] = 2 * 2 * 44100 * 15


def allwed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ファイルを受け取る方法の指定
@app.route('/', methods=['GET', 'POST'])
def uploads_file():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        # データの取り出し
        file = request.files['file']
        # ファイル名がなかった時の処理
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
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

            # アップロード後のページに転送
            filename_png= os.path.splitext(os.path.basename(filename))[0]+'.png'
            if rt_code:
                rt_codes=1
            else:
                rt_codes=0
            return redirect(url_for('result_file', filename=filename_png, t_time=t_time, rt_code=rt_codes))
    return render_template( "index.html" )


@app.route('/result', methods=["GET"])
# 推定結果を表示する
def result_file():
    #
    req = request.args
    filename = req.get("filename")
    t_time = req.get("t_time")
    rt_code= req.get("rt_code")
    return render_template("result.html", figure=os.path.join(app.config['RESULT_FOLDER'],filename), t_time=t_time, rt_code=rt_code)

if __name__ == "__main__":
    # 時間差を推定するインスタンスを作成する
    tde= time_difference_estimation('sample_wav/chime_only.wav', 1, 0,save_dir='static/figure/',SHOW_PLOT=False, ShowOntheWay=False, SHOW_PLOT2=False)
    # サーバーを起動する
    app.run(threaded=True)  # port=5000