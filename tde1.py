#coding:utf-8

#  (概要）
#  チャイム音をマイクでステレオ録音しとき、L channelとR channelの音の時間差を推定するもの。
#
#  (手順)
#  大枠を見積もるためchime_only.wavを読み込み　粗雑な極大点のエンベロープする。 
#  注意：基準となるchime_only.wavの先頭の無音区間を短めにしておくこと
#  （１）大雑把に相互相関を計算して類似箇所を見つける。
#　（２-0）類似箇所のスペクトログラムを計算し、スペクトログラムが変化の大きい位置を見つける。
#  （２-1）類似箇所のスペクトログラムを計算し、境界として期待される図形をあてて境界を探す。
#  （３）更に、ＬＰＦ／ＨＰＦを掛けた波形の中で注目する短時間の波形の相互相関を計算してL channelとR channelの時間差を予測する。
#

# Check version
# Python 3.6.4, 64bit on Win32 (Windows 10)
# numpy 1.18.4
# scipy 1.4.1
# matplotlib 3.3.1


import sys
import os
import glob
import argparse
import pathlib
import math
import copy

import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy import signal # version > 1.2.0
from scipy.signal import hilbert
from scipy.signal import find_peaks
from scipy import interpolate

from matplotlib import pyplot as plt


class time_difference_estimation(object):
    def __init__(self,path_clime_wav, wf_channel, tv_channel, save_dir='figure/', SHOW_PLOT=True, ShowOntheWay=True, SHOW_PLOT2=True):
        # initalize
        self.path_clime_wav= path_clime_wav
        self.wf_channel= wf_channel
        self.tv_channel= tv_channel
        self.SHOW_PLOT= SHOW_PLOT
        self.ShowOntheWay= ShowOntheWay  # 途中経過の表示のON/OFF
        self.SHOW_PLOT2= SHOW_PLOT2
        self.save_dir= save_dir
        if not os.path.exists( self.save_dir):
            os.mkdir( self.save_dir)
        self.test_number=0
        
        # chime wavを読み込む
        self.chime_wav, self.sr_chime= self.read_wav(  path_clime_wav)  #args.dir + args.clime_wav)
        print ('chime wav is ', path_clime_wav, self.sr_chime)
        #print ('amax', np.amax(chime_wav) )
        if self.chime_wav.shape[1] != 2:
            print ('error: chime wav is not stereo')
            sys.exit()
        
        # chime wavの先頭からの有効な長さ
        effect_time=1.5  #0.75 # unit second
        if self.chime_wav.shape[0] < int(self.sr_chime * effect_time):
            print ('error: chime wav is not enough length')
            sys.exit()
        
        self.chime_wav_portion=self.chime_wav[0: int(self.sr_chime * effect_time), self.wf_channel]
        self.chime_wav_crude_curve= self.crude_curve (self.chime_wav_portion, title='chime_wav', SHOW_PLOT=self.SHOW_PLOT)
        
        # spectrogramの隣接フレームの差が一番大きな位置の時間（フレーム単位）を取得する
        self.chime_wav_tpoint, rt_code =self.spectrogram2(self.chime_wav_portion, sr=self.sr_chime, title='chime_wav', SHOW_PLOT=self.SHOW_PLOT)
    
    
    def main0(self,file_path):
        #
        # 試験用のwavを読み込む
        w_ref0,sr= self.read_wav( file_path)
        print (file_path)
        #print ('amax', np.amax(w_ref0[:,0]), np.amax(w_ref0[:,1]) )
        save_path= self.save_dir + os.path.splitext(os.path.basename(file_path))[0] + '.png'
        fname= self.save_dir + 'output.txt'
        
        if sr != self.sr_chime:
            print ('error: sr is different.')
            return False
        if w_ref0.shape[1] != 2:
            print ('error: test wav is not stereo')
            return False
        
        ### wf ###
        wf_target_estimation1, wf_rt_code = self.sub_main1(w_ref0, sr, self.wf_channel, self.chime_wav_crude_curve, title='wf', SHOW_PLOT=self.ShowOntheWay)
        
        ### tv ###
        tv_target_estimation1, tv_rt_code = self.sub_main1(w_ref0, sr, self.tv_channel, self.chime_wav_crude_curve, title='tv', SHOW_PLOT=self.ShowOntheWay)
        
        # check if fault ?
        if (not wf_rt_code) or (not tv_rt_code):
            print('There is fault in estimation1 stage')
            return False
        ### comp both
        if self.tv_channel == 0:
            LR_estimation1= self.sub_main2(w_ref0, sr, tv_target_estimation1, wf_target_estimation1, save_path, title=file_path, SHOW_PLOT=self.ShowOntheWay, SHOW_PLOT2=self.SHOW_PLOT2)
        else:
            LR_estimation1= self.sub_main2(w_ref0, sr, wf_target_estimation1, tv_target_estimation1, save_path, title=file_path, SHOW_PLOT=ShowOntheWay, SHOW_PLOT2=self.SHOW_PLOT2)
        
        ### write to text file
        if self.test_number == 0: # new
            self.write_text(file_path, LR_estimation1, fname=fname, mode='w')
        else:    # append 
            self.write_text(file_path, LR_estimation1, fname=fname, mode='a')
        
        #
        self.test_number=self.test_number+1  # count +1
        return True  # estimation was done.
    
    
    def sub_main1(self, w_ref0, sr, channelx, chime_wav_crude_curve,title=None, SHOW_PLOT=True):
        #
        # chime_only.wavの粗雑な極大点のエンベロープとの相関による大雑把に対象となる位置を見つける
        crude_curve_out= self.crude_curve (w_ref0[:, channelx])
        cor,diff= self.corr2(crude_curve_out, chime_wav_crude_curve, sr=sr, title=title, SHOW_PLOT=SHOW_PLOT)
        
        # diffが複数ある場合は最初のピークから順番に試してみる。
        for i in range( len(diff)):
            # 目標位置の最初の予測時間
            target_estimation0= diff[i]+self.chime_wav_tpoint
            print ('target_estimation0', target_estimation0)
            
            # 波形表示のためのパラメーターの設定
            MIN_SPAN=  0.1  # 最小の時間幅  単位　秒　unit second
            
            sp0=target_estimation0- MIN_SPAN/2
            ep0=target_estimation0+ MIN_SPAN/2
            
            # 最初の予測時間の前後MIN_SPAN
            signalin= w_ref0[int(sp0 * sr): int(ep0 * sr),channelx] 
            
            if 1:
    	        # 最初の予測時間の前後MIN_SPANのspectrogramの
                # 境界として期待される図形をあてて推測した　境界の時間を取得する
                tpoint, rt_code =self.spectrogram2(signalin, sr=sr, t_point_mode=1, title=title, SHOW_PLOT=SHOW_PLOT)
            else:
                # 最初の予測時間の前後MIN_SPANのspectrogramの隣接フレームの差が一番大きな位置の時間（フレーム単位）を取得する
                tpoint, rt_code =self.spectrogram2(signalin, sr=sr, t_point_mode=0, title=title, SHOW_PLOT=SHOW_PLOT)
            # spectrogram2の結果がfalutでピークの候補がまだあれば、次の候補を試してみる。
            if rt_code:
                break
            else:
                print('re-try to spectrogram2', i)
        
        # 目標位置の最初の予測時間
        target_estimation1= tpoint+sp0
        print ('target_estimation1', target_estimation1)
        
        return target_estimation1, rt_code
    
    
    def sub_main2(self, w_ref0, sr, L_target_estimation1, R_target_estimation1,save_path, title=None, SHOW_PLOT=True, SHOW_PLOT2=True):
        #                     L channel[0] estimation, R channel[1] estimation
        # 波形表示のためのパラメーターの設定
        MIN_SPAN=  0.02  # 表示の時間幅  単位　秒　unit second
        
        L_sp0=L_target_estimation1- MIN_SPAN/2
        L_ep0=L_target_estimation1+ MIN_SPAN/2
        R_sp0=R_target_estimation1- MIN_SPAN/2
        R_ep0=R_target_estimation1+ MIN_SPAN/2
        sp0=min(L_sp0, R_sp0)
        ep0=max(L_ep0, R_ep0)
        L_signalin= w_ref0[int(sp0 * sr): int(ep0 * sr),0]
        R_signalin= w_ref0[int(sp0 * sr): int(ep0 * sr),1]
        L_x_time= np.linspace(0, len(L_signalin)/sr, len(L_signalin)) +sp0
        R_x_time= np.linspace(0, len(R_signalin)/sr, len(R_signalin)) +sp0
        
        # apply hpf/lpf
        w_ref=self.hpf_lpf(w_ref0, sr, f_center=450)
        L_signalin_hpf_lpf= w_ref[int(sp0 * sr): int(ep0 * sr),0] 
        R_signalin_hpf_lpf= w_ref[int(sp0 * sr): int(ep0 * sr),1] 
        
        # 波形の注目する部分のパラメーターの設定
        MIN_SPAN2=  0.010  # 注目する時間幅 < 表示の時間幅 <  単位　秒　unit second
        L_signalin2=L_signalin_hpf_lpf.copy()
        L_signalin2[: int( (L_target_estimation1 - MIN_SPAN2/2- sp0)  * sr)] =0
        L_signalin2[int( (L_target_estimation1+ MIN_SPAN2/2 - sp0)  * sr):] =0
        R_signalin2=R_signalin_hpf_lpf.copy()
        R_signalin2[: int( (R_target_estimation1 - MIN_SPAN2/2- sp0)  * sr)] =0
        R_signalin2[int( (R_target_estimation1+ MIN_SPAN2/2 - sp0)  * sr):] =0
        
        #
        cor0= signal.correlate(L_signalin2,R_signalin2)
        diff0= np.argmax(cor0) - len(L_signalin2)
        #print('diff0/sr', diff0 / sr)
        # 符号がマイナスの場合
        if diff0 < 0:
            cor0inv=cor0[::-1]
            cor0half= cor0inv[len(L_signalin2):]
        else:
            cor0half= cor0[len(L_signalin2):]
        
        if SHOW_PLOT:
            fig = plt.figure()
            ax1 = fig.add_subplot(4, 1, 1)
            ax2 = fig.add_subplot(4, 1, 2)
            ax3 = fig.add_subplot(4, 1, 3)
            ax4 = fig.add_subplot(4, 1, 4)
            
            ax1.plot(L_x_time, L_signalin, color='blue') 
            ax1.plot(R_x_time, R_signalin, color='red') 
            ax1.plot(L_target_estimation1, L_signalin[ int( (L_target_estimation1 - sp0)* sr)], "x",color='blue')
            ax1.plot(R_target_estimation1, R_signalin[ int( (R_target_estimation1 - sp0)* sr)], "o", color='red')
            ax1.grid(which='both', axis='both')
            if title is not None:
                ax1.set_title(title)
            
            #
            ax2.plot(L_x_time, L_signalin2, color='blue') 
            ax2.plot(R_x_time, R_signalin2, color='red') 
            ax2.grid(which='both', axis='both')
            
            #
            x_time2= np.linspace(0, len(cor0half)/sr, len(cor0half))
            ax3.plot(x_time2, cor0half ,color='blue' )
            # 符号がマイナスの場合もある
            ax3.plot( abs(diff0) / sr, cor0half[ int(abs(diff0))], "x",color='blue')
            ax3.set_title( str(diff0 / sr))
            ax3.grid(which='both', axis='both')
            
            #
            ax4.plot(L_x_time, L_signalin2, color='blue') 
            ax4.plot(R_x_time, R_signalin2, color='red') 
            # 符号がマイナスの場合
            if diff0 < 0:
                ax4.plot(L_x_time- diff0 / sr, L_signalin2, color='yellow') 
            else:
                ax4.plot(R_x_time+ diff0 / sr, R_signalin2, color='yellow') 
            ax4.grid(which='both', axis='both')
            
            plt.tight_layout()
            plt.show()
            plt.close()
        
        #if SHOW_PLOT2:
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax1.plot(L_x_time, L_signalin, color='blue') 
        ax1.plot(R_x_time, R_signalin, color='red') 
        ax1.grid(which='both', axis='both')
        if title is not None:
            ax1.set_title(title)
        
        # 符号がマイナスの場合
        if diff0 < 0:
            ax2.plot(L_x_time- diff0 / sr, L_signalin, color='yellow') 
            ax2.plot(R_x_time, R_signalin, color='red') 
        else:
            ax2.plot(R_x_time+ diff0 / sr, R_signalin, color='yellow') 
            ax2.plot(L_x_time, L_signalin, color='blue') 
        
        ax2.set_title('estimation time ' +  str(diff0 / sr))
        ax2.grid(which='both', axis='both')
        
        plt.tight_layout()
        if SHOW_PLOT2:
        	plt.show()
        
        # always save as a figure
        fig.savefig(save_path)
        
        plt.close()
            
        print ('LR_estimation1', diff0/sr)
        
        return diff0/sr #  予想される L R の時間差を返す
    
    
    def spectrogram2(self, signalin1, sr=44100, t_point_mode=0, title=None, SHOW_PLOT=True):
        # スペクトログラムの計算のためのパラメーターの設定
        NFFT=256    # 1スパンサイズ
        NOVERLAP=int(NFFT * 0.8)  # オーバーラップサイズ量（シフト量とは違う）
        YLIMH=5000  # 表示の上限周波数
        YLIML=100   # 表示の下限周波数
        YLIMH2=5000  # 比較のための上限周波数
        YLIML2=100   # 比較のための下限周波数
        
        signalin= signalin1
        
        fig = plt.figure()
        ax1 = fig.add_subplot(5, 1, 1)
        ax2 = fig.add_subplot(5, 1, 2)
        ax3 = fig.add_subplot(5, 1, 3)
        ax4 = fig.add_subplot(5, 1, 4)
        ax5 = fig.add_subplot(5, 1, 5)
        
        # spectrogramを計算する
        Pxxl, freqsl, binsl, iml = ax2.specgram(signalin, NFFT=NFFT, Fs=sr, noverlap=NOVERLAP)
        
        low_index= np.amin(np.where( freqsl > YLIML2))
        high_index= np.amax(np.where( freqsl < YLIMH2))
        specl= np.log10(Pxxl[low_index:high_index,:]) *10
        
        # 特徴を捉えるため最大値から閾値以下のものは背景とみなし一定の値に置き換える
        DYNAMIC_RANGE=20
        specl_max= np.amax(specl)
        specl_min= np.amin(specl)
        specl_drange= np.where( specl < (specl_max-DYNAMIC_RANGE), specl_min, specl)
        
        # spectrogramの隣接フレームの差が一番大きな位置を得る
        # 有音の部分などの　非局所的な情報も加味する必要あり　！？
        #-----------------------------------------------------------------------
        spec_diffw, peaksw, t_pointw = self.spec_diff_x(specl,binsl)
        spec_diffw_dr, peaksw_dr, t_pointw_dr = self.spec_diff_x(specl_drange,binsl)
        #print ('t_pointw', t_pointw)
        
        # 境界として期待される図形をあてて境界を探す
        cor0x, t_pointb, bins_cor0x =self.pick_boundary(specl_drange, binsl)
        
        # ピーク最小幅  このチューニングだけでは動作は完璧にならないね。
        MIN_HIGH= 0.4   # ピークの最小高さ
        MIN_DIS= 0.1    # 最小の周辺距離
        MIN_WIDTH= 0.03 # 最小の周辺幅
        MIN_RATIO= 5    # 富士山の様なピークが　他のピークより　何倍以上大きいか
        
        peaks, _ = signal.find_peaks(cor0x, height= MIN_HIGH * max(cor0x), distance= MIN_DIS * cor0x.shape[0] , width= MIN_WIDTH * cor0x.shape[0])
        
        rt_code=True
        # 富士山の様なピークでない場合は　上手く行っていない
        if len(peaks) >= 2:  # ピークが2個以上ある
            peak_value=sorted( cor0x[peaks], reverse=True)
            print ('sort of cor0x peaks', sorted( cor0x[peaks], reverse=True))
            if peak_value[0] < (peak_value[1] * MIN_RATIO):
                print('fault: it is not like a peak as Mt Fuji.')
                rt_code= False # return value as fault
        
        if SHOW_PLOT:
            x_time= np.linspace(0, len(signalin)/sr, len(signalin)) 
            ax1.plot(x_time, signalin)
            ax1.grid(which='both', axis='both')
            if title is not None:
                ax1.set_title(title)
            
            fig.colorbar(iml, ax=ax2).set_label('Intensity [dB]')
            ax2.grid(which='both', axis='both')
            ax2.set_ylim(YLIML,YLIMH)
            
            
            extent0=[0, binsl[-1], freqsl[low_index], freqsl[high_index]]
            ax3.imshow(specl_drange, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest', extent=extent0)
            ax3.grid(which='both', axis='both')
            
            ax4.plot(binsl[:-1],spec_diffw, color='blue')
            ax4.plot(binsl[:-1],spec_diffw_dr, color='red')
            ax4.grid(which='both', axis='both')
            
            ax5.plot(bins_cor0x, cor0x)
            ax5.plot(bins_cor0x[peaks], cor0x[peaks], "x")
            ax5.grid(which='both', axis='both')
            
            if t_point_mode==0:
                ax1.plot(t_pointw, signalin[ int(t_pointw * sr)], "x")
            elif t_point_mode==1:
                ax1.plot(t_pointb, signalin[ int(t_pointb * sr)], "x")
            
            plt.tight_layout()
            plt.show()
        plt.close()
        
        if t_point_mode==0:
            # spectrogramの隣接フレームの差が一番大きな位置の時間を返す（フレーム単位）
            return t_pointw, rt_code
        elif t_point_mode==1:
            #  spectrogramの最大値から閾値以下のものは背景とみなし一定の値に置き換えたものに
            # 境界として期待される図形をあてて推測した　境界の時間を返す
            return t_pointb, rt_code
        else:
            return -1 # return negative value as error
    
    
    def crude_curve(self, signalin, title=None, SHOW_PLOT=False):
        # 粗雑な極大点のエンベロープを求める
        # find_peaks のdistanceの長さ
        effect_distance=0.002 #unit second
        
        signalw = hilbert(signalin)
        envelopew = np.abs(signalw)
        
        peaks, _ = find_peaks(envelopew, distance=max(1,int(effect_distance * self.sr_chime)))
        x_time= np.linspace(0, len(signalin)/self.sr_chime, len(signalin))
        n_time= np.linspace(0, len(signalin), len(signalin))
        
        y=np.concatenate([ [envelopew[0]], envelopew[peaks], [envelopew[-1]]  ])
        x=np.concatenate([ [0], peaks, [int(len(envelopew)-1)]  ])
        
        ###crude_curve = interpolate.interp1d(x, y, kind = 'cubic') # return function
        crude_curve = interpolate.CubicSpline(x, y)
        crude_curve_out= crude_curve( n_time )
        
        # 極大点をピックアップする
        localmaxid = signal.argrelmax(crude_curve_out)
        y=np.concatenate([ [crude_curve_out[0]], crude_curve_out[localmaxid[0]], [crude_curve_out[-1]]  ])
        x=np.concatenate([ [0], localmaxid[0] , [int(len(envelopew)-1)]  ])
        crude_curve2 = interpolate.CubicSpline(x, y)
        crude_curve_out2= crude_curve2( n_time )
        
        if SHOW_PLOT:
            fig = plt.figure()
            ax1 = fig.add_subplot(3, 1, 1)
            ax2 = fig.add_subplot(3, 1, 2)
            ax3 = fig.add_subplot(3, 1, 3)
            
            ax1.plot( x_time, signalin, color='blue') 
            ax1.plot( x_time, envelopew ,  color='yellow', label='envlop ref') 
            ax1.grid(which='both', axis='both')
            if title is not None:
                ax1.set_title(title)
            
            ax2.plot( x_time, envelopew, color='blue') 
            ax2.plot(x_time[peaks], envelopew[peaks], "x")
            ax2.grid(which='both', axis='both')
            
            ax3.plot( x_time, envelopew, color='blue') 
            ax3.plot( x_time, crude_curve_out, color='red',)
            ax3.plot( x_time[localmaxid[0]], crude_curve_out[localmaxid[0]], "x")
            ax3.plot( x_time, crude_curve_out2, color='yellow',)
            ax3.grid(which='both', axis='both')
            
            plt.tight_layout()
            plt.show()
            plt.close()
        
        return crude_curve_out2 #　粗雑な極大点のエンベロープを返す
    
    
    def corr2(self, signalin1, signalin2ref, sr=44100, title=None, SHOW_PLOT=True):
        # 相関を計算して signalin2refと類似した、signalin1の中の位置を予想する
        # 長さを揃える
        if len(signalin2ref) < len(signalin1):
            signalin2a= np.append(signalin2ref, np.zeros(len(signalin1)-len(signalin2ref)))
            # Lch と chime.wav の相関を計算する
            cor0= signal.correlate(signalin1,signalin2a)
        else:
            print ('error: len(signalin2ref) > len(signalin1)')
            cor0=NULL
            diff0=0
            return cor0, diff0
        
        # ピーク最小幅  このチューニングだけでは動作は完璧にならないね。
        MIN_HIGH= 0.4   # ピークの最小高さ
        MIN_DIS= 0.1    # 最小の周辺距離
        MIN_WIDTH= 0.03 # 最小の周辺幅
        
        peaks, _ = signal.find_peaks(cor0, height= MIN_HIGH * max(cor0), distance= MIN_DIS * cor0.shape[0] , width= MIN_WIDTH * cor0.shape[0])
        
        # ピークの系列を返す
        RETURN_PEAKS=True
        
        if RETURN_PEAKS:
            diff0 = peaks - len(signalin1)
        else: #RETURN_MAX_PEAK:
            diff0= [ np.argmax(cor0) - len(signalin1) ]
        
        if SHOW_PLOT:
            fig = plt.figure()
            ax1 = fig.add_subplot(3, 1, 1)
            ax2 = fig.add_subplot(3, 1, 2)
            ax3 = fig.add_subplot(3, 1, 3)
            
            x_time= np.linspace(0, len(signalin1)/sr, len(signalin1))
            x_time2= np.linspace(0, len(signalin2ref)/sr, len(signalin2ref))
            ax1.plot(x_time, signalin1, color='blue') 
            ax1.grid(which='both', axis='both')
            if title is not None:
                ax1.set_title(title)
            
            ax2.plot(x_time, signalin2a, color='blue') 
            ax2.plot(x_time2+diff0[0] / sr, signalin2ref, color='yellow') 
            ax2.grid(which='both', axis='both')
            
            x_time3= np.linspace(0, len(cor0)/sr, len(cor0))
            #print ('np.argmax(cor0)',  np.argmax(cor0) / sr)
            #print ('diff0',  diff0 / sr)
            ax3.plot( x_time3, cor0)
            ax3.plot( x_time3[peaks], cor0[peaks], "x")
            ax3.grid(which='both', axis='both')
            
            plt.tight_layout()
            plt.show()
            plt.close()
        
        return cor0, diff0 / sr  #  予想される 位置の時間を返す
    
    
    def pick_boundary(self, in_box, bins):
        #
        # 左側がー１、右側が＋１の図形をあてて、境界を探す。
        box_size=10
        r_box0= np.ones((in_box.shape[0],box_size))*-1
        r_box1= np.ones((in_box.shape[0],box_size))
        r_box=  np.concatenate([r_box0, r_box1],axis=1)
        
        cor0= signal.correlate2d(in_box, r_box)
        cor0x= cor0[ int(cor0.shape[0]/2),int(2 * box_size): int(-1 * 2 * box_size)]  # 前後 2*box_sizeの部分は削除する
        idmax=np.argmax(cor0x) #  最大変化位置
        
        if 0:
            print ('in_box.shape', in_box.shape)
            print ('bins.shape', bins.shape)
            print('r_box.shape', r_box.shape)
            print('cor0.shape', cor0.shape)
            #print('cor0', cor0[ int(cor0.shape[0]/2),:]  )
            print('cor0x.shape', cor0x.shape)
            #  bins[idmax+ box_size]の時間が境界に相当か？
            #  cor0 の先頭のr_box分を無視して計算してよいのか不明？
            print('idmax', idmax, bins[idmax], bins[idmax+ box_size]) 
        
        return cor0x, bins[idmax+ box_size], bins[box_size: cor0x.shape[0]+ box_size]
    
    
    def spec_diff_x(self, spec, bins=None):
        iy=spec.shape[1]  # time
        ix=spec.shape[0]  # 周波数軸
        
        spec_diff= np.abs(spec[:,1:iy] - spec[:,0:iy-1])  # 差の絶対値
        ###spec_diff= spec[:,1:iy] - spec[:,0:iy-1]  # 差
        spec_diff_mean= np.mean( spec_diff,axis=0)        # 周波数軸に沿って平均値
        
        # ピーク最小幅
        MIN_WIDTH= 9    # 最小の時間幅　ｍＳ単位
        MIN_HIGH= 0.7   # ピークの最小高さ
        
        peaks, _ = signal.find_peaks(spec_diff_mean, height= MIN_HIGH * max(spec_diff_mean) )
        
        idmax=np.argmax(spec_diff_mean[peaks]) #  最大変化位置
        
        
        if bins is not None:
            t_point= bins[peaks[idmax]+1] # 変化後のためプラス１する 
            return spec_diff_mean, peaks, t_point
        else:
            return spec_diff_mean, peaks # 差をとるので次数が1個減る
    
    
    def hpf_lpf(self, w_ref, sr, f_center=450):
        # apply hpf and lpf
        hb, ha = signal.iirfilter(4, f_center ,  btype='highpass', ftype='butter', fs=sr )
        lb, la = signal.iirfilter(4, f_center ,  btype='lowpass', ftype='butter', fs=sr )
        y=signal.lfilter(hb, ha, w_ref[:,0])
        y0=signal.lfilter(lb, la, y)
        y=signal.lfilter(hb, ha, w_ref[:,1])
        y1=signal.lfilter(lb, la, y)
        return np.stack([y0,y1],axis=1)
    
    
    def read_wav(self, file_path ):
        try:
            sr, w = wavread( file_path)
        except:
            print ('error: wavread ', file_path)
            sys.exit()
        else:
            if w.dtype ==  np.int16:
                #print('np.int16')
                w= w / (2 ** 15)
            elif w.dtype ==  np.int32:
                #print('np.int32')
                w= w / (2 ** 31)
            #print ('sampling rate ', sr)
            #print ('size', w.shape) # [xxx,2]
        return w, sr
    
    
    def save_wav(self,  file_path, data, sr=48000):
        amplitude = np.iinfo(np.int16).max
        try:
            wavwrite( file_path , sr, np.array( amplitude * data , dtype=np.int16))
        except:
            print ('error: wavwrite ', file_path)
            sys.exit()
        print ('wrote ', file_path)
    
    
    def write_text(self, file_path, value, fname='output.txt',mode='a'):
        with open( fname, mode, encoding='UTF-8') as f:
            f.write(file_path)
            f.write(',')
            f.write(str(value))
            f.write('\n')
            f.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='estimate L and  R signal difference time')
    parser.add_argument('--wf_channel', '-r', type=int, default=1, help='specify wf channel of L(0) R(1)')
    parser.add_argument('--dir', '-d', default='WAV/', help='specify test wav directory')
    parser.add_argument('--tv_channel', '-t', type=int, default=0, help='specify tv channel L(0) R(1)')
    parser.add_argument('--clime_wav', '-c', default='sample_wav/chime_only.wav', help='specify chime wav as reference')
    args = parser.parse_args()
    
    # trial some test wav files
    if 1:
        flist= \
        [
         "sample_wav/1.wav",  
         "sample_wav/71.wav",
         "sample_wav/100.wav",
        ]
    else: # 指定ホルダーの中にあるwavを計算する
        flist= glob.glob( args.dir + '*.wav')
    
    # chime wavのインスタンスを生成
    SHOW=False #　波形グラフを表示するかしないかを設定
    tde= time_difference_estimation(args.clime_wav, args.wf_channel, args.tv_channel,SHOW_PLOT=SHOW, ShowOntheWay=SHOW, SHOW_PLOT2=SHOW)
    
    
    # 試験用のwavを読ませて時間差を推定してみる
    for i,file_path in enumerate(flist):
        tde.main0( file_path)