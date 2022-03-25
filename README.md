#  Wav Input, Time Difference Estimation Output, Web　App    

チャイム音をマイクでステレオ録音したWAVファイルを入力して、L channelとR channelの音の時間差をpythonを使って推定して、その結果画像を表示するFlaskを使ったWeb　App。  



## 内容   

tde1.py チャイム音をマイクでステレオ録音しとき、L channelとR channelの音の時間差を推定するためのクラス。  
sample_wav チャイム音をマイクでステレオ録音したwavファイルのサンプルが入っている。  
uploads　入力したWAVファイルが入る。  
figure 結果画像が入る。推定時間を記述したテキストファイルも入る。  
  
## 実験  
  
google colabとngorkをつかって動作確認するためのスクリプト
[API_server_trial1.ipynb](https://colab.research.google.com/github/shun60s/time-difference-WebApp/blob/master/API_server_trial1.ipynb)  



