# -*- coding: utf-8 -*-


"""
チャイム音をマイクでステレオ録音したWAVファイルを入力して、L channelとR channelの音の時間差をpythonを使って推定して、
その結果画像を表示するFlaskとajaxを使ったWeb Appの動作実験するもの。

DATABASEへの動作内容の書き込みを追加したもの。
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
import base64
import glob

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

# アップロードと結果画像を削除するかどうかのフラグ
REMOVE_FLAG = True

app = Flask(__name__)  #, static_folder=RESULT_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['BLANK_FIGURE'] = BLANK_FIGURE
app.config['ERROR_FIGURE'] = ERROR_FIGURE

# アップロードされる容量を制限する
app.config['MAX_CONTENT_LENGTH'] = 2 * 2 * 44100 * 15


# 時間差を推定するインスタンスを作成する
tde= time_difference_estimation('sample_wav/chime_only.wav', 1, 0,save_dir='static/figure/',SHOW_PLOT=False, ShowOntheWay=False, SHOW_PLOT2=False)


#------ DATABASE 書き込み関連 -------
import psycopg2
from psycopg2 import Error
import pandas as pd
import pandas.io.sql as psql
import sqlite3
import datetime

# DATABASE URLが環境変数として定義されているか確認する。
try:
    DATABASE_URL = os.environ['DATABASE_URL']
    print ('DATABASE_URL ', DATABASE_URL)
    DB_NAME = 'output_4'  # テーブルの名前を設定する
except KeyError:
    DATABASE_URL=None
    print ('There is no DATABASE_URL')

# DATABASEに接続できるか試してみる
try:
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    print ('connect database ', DATABASE_URL)
except(Exception, Error) as error:
    DATABASE_URL=None
    print ('error: cannot connect database')
    

# 以下、DATABASEとの通信にタイムアウトがないのは要改善点

def insert_to_db(filename, t_time, rt_code):
    # insert to database
    if DATABASE_URL is not None:
        uuid_str= str(uuid.uuid1())
        dt_now = datetime.datetime.now()
        dt_now_str= dt_now.isoformat()
        df=pd.DataFrame( [[uuid_str, dt_now_str, filename, t_time, rt_code]], columns=[ 'uuid', 'dt_now', 'filename', 't_time','rt_code'] )
        
        with sqlite3.connect("output.db") as conn:
            psql.to_sql( df, DB_NAME , conn , if_exists='append', index=False)


def read_from_db():
    # read from  database
    if DATABASE_URL is not None:
        with sqlite3.connect("output.db") as conn:
            db = conn.execute("select dt_now, filename, t_time, rt_code from " + DB_NAME )
            db_content= db.fetchall()
            #print( db.fetchall())
            db.close()
        return db_content
    else:
        return None


def delete_all():
    # delete all
    # idを指定して一行づつ削除していく
    if DATABASE_URL is not None:
        with sqlite3.connect("output.db") as conn:
            db = conn.execute("select uuid from " + DB_NAME )
            db_all=db.fetchall()
            for db1 in db_all: 
                db1_uuid = db1[0]
                print('db1_uuid', db1_uuid)
                db = conn.execute("delete from " + DB_NAME + "  where uuid=?", (db1_uuid,))
            db.close()


# 起動時にDATABASEの中身を消す場合は１にする。
if 0:
    delete_all()
#------ DATABASE 書き込み関連 -------




def allwed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# トップページ
@app.route("/")
def index():
    return render_template('index-ajax-b64.html', filename=app.config['BLANK_FIGURE'], filename2=app.config['ERROR_FIGURE'])

# ファイルを受け取る方法の指定
@app.route('/upload', methods=['POST'])
def upload_and_tde():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'file' not in request.files:
            return jsonify(t_time=0, rt_code=0)
        # データの取り出し
        file = request.files['file']
        # ファイル名がなかった時の処理
        if file.filename == '':
	        return jsonify(t_time=0, rt_code=0)
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
            # アップロードを削除するかどうか?
            if REMOVE_FLAG:
                os.remove(file_path)
            # 推定結果画像の場所を返す
            if rt_code:
                rt_codes=1
                filename=os.path.join(app.config['RESULT_FOLDER'],filename)
                filename_png= os.path.join(app.config['RESULT_FOLDER'], os.path.splitext(os.path.basename(filename))[0]+'.png')
                # 画像データをBase64に変換する
                result_fig_data= open(filename_png,"rb").read()
                result_fig_data_b64= base64.b64encode(result_fig_data).decode('utf-8')
                # 結果画像を削除するかどうか?
                if REMOVE_FLAG:
                    os.remove(filename_png)
                # DATABASEへ書き込み
                insert_to_db(filename, t_time, rt_code)
                return jsonify(t_time=t_time, rt_code=rt_code, figure_b64=result_fig_data_b64)
            else:  # tdeの中でエラー発生
                rt_codes=0
                # DATABASEへ書き込み
                insert_to_db(filename, 0, rt_code)
                return jsonify(t_time=0, rt_code=rt_code)
            
        else:
            return jsonify(t_time=0, rt_code=0)

# 管理用ページ
@app.route("/show")
def show_list():
    # uploadフォルダーの中にあるWAVファイルのリストと
    # resultフォルダーの中にあるpngファイルのリストを返す
    # http://127.0.0.1:5000/show
    upload_list = glob.glob( UPLOAD_FOLDER + '/*.wav')
    figure_list = glob.glob( RESULT_FOLDER + '/*.png')
    return render_template('show.html', upload_list=upload_list, figure_list=figure_list)

@app.route("/show_db")
def show_db():
    # DATABASEの中身を表示する
    # http://127.0.0.1:5000/show_db
    db_content= read_from_db()
    if db_content is not None:
        print ('len(db_content)',len(db_content))
        return render_template('show_db.html', database_contents=db_content)
    else:
        return render_template('show_db.html', database_contents=['None'])


if __name__ == "__main__":
    # アップロードと結果画像を削除するかどうかのフラグ
    # 削除しない場合は Falseを再設定する（初期値は Trueに設定している)
    REMOVE_FLAG = False
    # サーバーを起動する
    app.run(threaded=True) 
