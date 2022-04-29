#coding:utf-8

#  (概要）
#  チャイム音(chime_wav)をマイクでステレオ録音しとき、L channelとR channelの音の時間差を推定するもの。
#
#  (方法）
#　調波構造の majorな部分を抜きとって、再合成して　２つの信号を比較し、時間差を求める。
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
import datetime


import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy import signal # version > 1.2.0
from scipy.signal import hilbert
from scipy.signal import find_peaks

from matplotlib import pyplot as plt

from BPF import *


class time_difference_estimation(object):
    def __init__(self,path_clime_wav, ref_channel, another_channel=None, save_dir='figure/', SHOW_PLOT=True, ShowOntheWay=True, SHOW_PLOT2=True):
        # initalize
        self.path_clime_wav= path_clime_wav
        self.ref_channel= ref_channel
        self.another_channel= another_channel
        self.SHOW_PLOT= SHOW_PLOT
        self.SHOW_PLOT2= SHOW_PLOT2
        self.save_dir= save_dir
        if not os.path.exists( self.save_dir):
            os.mkdir( self.save_dir)
        self.test_number=0
        
        # BPFの個数
        self.NUM_BPF=4
        
        ##############################################
        # plot_wav2でのchime_wavの表示位置。 手動設定となっている。
        self.offsetx=0.305-0.05
        self.lng0=0.1
        ##############################################
        
        
        # when all not show, use non-GUI backend.
        if (not self.SHOW_PLOT) and (not self.SHOW_PLOT2):
            import matplotlib
            matplotlib.use('Agg')
        
        # chime wavを読み込む
        self.chime_wav, self.sr_chime= self.read_wav(  path_clime_wav)  #args.dir + args.clime_wav)
        print ('chime wav is ', path_clime_wav, self.sr_chime)
        #print ('amax', np.amax(chime_wav) )
        if self.chime_wav.shape[1] != 2:
            print ('error: chime wav is not stereo')
            sys.exit()
        
        #################################################################
        # chime wav の有効な長さにする。　試験サンプル中で有効な部分が一番短いものに合わせて調整する。
        self.effect_time=1.1 #1.15 #1.2 #1.5 #1.8
        if self.chime_wav.shape[0] < int(self.sr_chime * self.effect_time):
            print ('error: chime wav is not enough length')
            sys.exit()
        self.chime_wav = self.chime_wav[0: int(self.sr_chime * self.effect_time)]
        
        
        # chime wavの先頭からのFFTを計算する開始時刻と長さ
        self.start_time=0.4  # second
        self.fft_length= 4096 #2048 # point
        if self.chime_wav.shape[0] < (int(self.sr_chime * self.start_time) + self.fft_length ):
            print ('error: chime wav is not enough length for FFT')
            sys.exit()
        
        # PEAK周波数の候補を求める
        self.freq_list= self.sub_fft(SHOW_FREQ_BINS=True, SHOW_PLOT=self.SHOW_PLOT)
        
        # BPFを作成する
        self.F_LIST= self.make_bpf(self.freq_list, self.NUM_BPF)
        
        # BPFを掛けたものを合算したものを作る
        syth_out,_ , self.envelopew= self.apply_bpf(self.chime_wav[:,self.ref_channel],SHOW_PLOT=self.SHOW_PLOT)
        
        #
        self.show_douzisei(self.envelopew, SHOW_PLOT=self.SHOW_PLOT)
        
        
        # 2番目のチャンネルの状態も参照する。
        if another_channel is not None:
            syth_out_an,_, self.envelopew_an= self.apply_bpf(self.chime_wav[:,self.another_channel],title="another channel", SHOW_PLOT=self.SHOW_PLOT)
            # 比較してみる。
            self.show_envelopew(self.envelopew, self.envelopew_an, SHOW_PLOT=self.SHOW_PLOT)
        
        
        # WAVファイルとして書き出す
        if 0:
            file_path='syth_out.wav'
            self.save_wav(file_path, syth_out, sr=self.sr_chime)
        
        
    #     220,440,660,880
    #     330, 660
    #     
    def sub_fft(self, signalin=None, sr=None, title=None, SHOW_FREQ_BINS=False, SHOW_PLOT=False):
        # FFTする信号を設定する。
        if signalin is None:
            signalin= self.chime_wav[ int(self.sr_chime * self.start_time): int(self.sr_chime * self.start_time)+ self.fft_length, self.ref_channel]
        
        if sr is None:
             sr= self.sr_chime
        
        
        # FFT
        n=len(signalin)
        window = np.hamming(n)
        fftout = np.fft.fft(signalin * window)
        fftout_abs= np.abs(fftout)
        fftout_abs_log= np.log10( np.where(fftout_abs < 1.0e-15, -15, fftout_abs)) * 20
        freq_bins= np.fft.fftfreq( n, 1.0/ sr)
        
        x_time= np.linspace(0, n/sr, n)
        
        # 表示する最高周波数
        MAX_FREQ=3000
        high_index=np.where(freq_bins > MAX_FREQ)[0][0]
        
        # peak探索
        MIN_HIGH= 0.4   # ピークの最小高さ
        #MIN_DIS= 0.000001    # 最小の周辺距離
        MIN_WIDTH= 2 # 最小の周辺幅
        portion1= fftout_abs_log[1: high_index]
        freq_bins_portion1= freq_bins[1: high_index]
        peaks, _ = signal.find_peaks(portion1, height= MIN_HIGH * max(portion1),width= MIN_WIDTH) #, distance= MIN_DIS * n , width= MIN_WIDTH * n)
        
        # try to pick up harmonic structure
        # peakを大きいもの順に並べる
        if SHOW_FREQ_BINS:
            print('peak frequency list:')
            print(freq_bins_portion1[peaks])
        print('index, frequency, value[dB]:')
        freq_list=[]
        for i in range(len(portion1[peaks])):
            freq0=freq_bins_portion1[peaks][np.argsort(portion1[peaks])[::-1][i]]
            freq_list.append(freq0)
            if SHOW_FREQ_BINS:
                print ( np.argsort(portion1[peaks])[::-1][i], freq0, np.sort(portion1[peaks])[::-1][i])
        
        
        if SHOW_PLOT:
            fig = plt.figure()
            ax1 = fig.add_subplot(3, 1, 1)
            ax2 = fig.add_subplot(3, 1, 2)
            ax3 = fig.add_subplot(3, 1, 3)
            
            ax1.plot( x_time, signalin, color='blue') 
            ax1.grid(which='both', axis='both')
            if title is not None:
                ax1.set_title(title)
            
            # freq_bins[0]のDC成分は除く
            ax2.plot(freq_bins[1: high_index],fftout_abs_log[1: high_index], color='blue') 
            ax2.plot(freq_bins_portion1[peaks], portion1[peaks], "x")
            ax2.grid(which='both', axis='both')
            
            
            plt.tight_layout()
            plt.show()
            plt.close()
        
        return freq_list
    
    def make_bpf(self,freq_list, NUM_BPF, sr=None):
        #
        if sr is None:
            sr=  self.sr_chime
            
        # instance
        self.bpf_list=[]
        # BPFの個数
        #NUM_BPF=NUM
        F_LIST=[]
        for i in range( min(NUM_BPF, len(freq_list))):
            self.bpf_list.append(Class_BPF(fc=freq_list[i],  Q=20.0, sampling_rate=sr))
            F_LIST.append( str(int(freq_list[i])) )
        
        return F_LIST
    
    def apply_bpf(self,signalin, title=None, SHOW_PLOT=False):
        # instance
        bpf_out=[]
        envelopew=[]
        syth_out=np.zeros(len(signalin))
        for i in range(len(self.bpf_list)):
            #########################################################
            # チャイム音と人の声と区別できないので、BPFの特性を急進にするためＢＰＦを2回かけてみる。
            if 1:
                signalout= self.bpf_list[i].filtering(signalin)
                bpf_out.append(self.bpf_list[i].filtering(signalout))
            else:
                bpf_out.append(self.bpf_list[i].filtering(signalin))
            #########################################################
            syth_out= syth_out + bpf_out[i]
            
            # エンベロープを求める
            signalw = hilbert(bpf_out[i])
            envelopew.append(np.abs(signalw))
            
        
        if SHOW_PLOT:
            fig = plt.figure()
            ax1 = fig.add_subplot(5, 1, 1)
            ax2 = fig.add_subplot(5, 1, 2)
            ax3 = fig.add_subplot(5, 1, 3)
            ax4 = fig.add_subplot(5, 1, 4)
            ax5 = fig.add_subplot(5, 1, 5)
            
            ax1.plot(bpf_out[0], color='blue') 
            ax1.plot(envelopew[0], color='red') 
            ax1.grid(which='both', axis='both')
            if title is not None:
                ax1.set_title(title)
            
            ax2.plot(bpf_out[1], color='blue') 
            ax2.plot(envelopew[1], color='red') 
            ax2.grid(which='both', axis='both')
            
            ax3.plot(bpf_out[2], color='blue') 
            ax3.plot(envelopew[2], color='red') 
            ax3.grid(which='both', axis='both')
            
            ax4.plot(bpf_out[3], color='blue') 
            ax4.plot(envelopew[3], color='red') 
            ax4.grid(which='both', axis='both')
            
            ############################
            if 0:
                ax5.plot(syth_out, color='blue') 
                ax5.grid(which='both', axis='both')
            
            for i in range(4):
                ax5.plot(envelopew[i]) 
            ###########################
            
            
            plt.tight_layout()
            plt.show()
            plt.close()
        
        return syth_out,bpf_out,envelopew
    
    def show_envelopew(self, envelopew, envelopew_an, F_LIST=None, NUM=None, SHOW_PLOT=False):
        #
        if F_LIST is None:
            F_LIST= self.F_LIST
        if NUM is None:
            NUM=self.NUM_BPF
            
        x_time3= np.linspace(0, len(envelopew[0])/self.sr_chime, len(envelopew[0]))
        color_list = ["r", "g", "b", "c", "m", "y", "k", "w"]
        
        #　２つのenvelopewの比較
        if SHOW_PLOT:
            fig = plt.figure()
            ax1 = fig.add_subplot(3, 1, 1)
            ax2 = fig.add_subplot(3, 1, 2)
            ax3 = fig.add_subplot(3, 1, 3)
            
            
            for i in range(NUM):
                ax1.plot(x_time3, envelopew[i], color=color_list[i], label=F_LIST[i]) 
            
            ax1.grid(which='both', axis='both')
            ax1.legend()
            
            
            for i in range(NUM):
                ax2.plot(x_time3, envelopew_an[i], color=color_list[i], label=F_LIST[i]) 
            
            ax2.grid(which='both', axis='both')          
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            plt.close()
    
    def show_douzisei(self, envelopew, F_LIST=None, NUM=None, title=None, SHOW_PLOT=False):
    	#
        if F_LIST is None:
            F_LIST= self.F_LIST
        if NUM is None:
            NUM=self.NUM_BPF
        
        x_time3= np.linspace(0, len(envelopew[0])/self.sr_chime, len(envelopew[0]))
        color_list = ["r", "g", "b", "c", "m", "y", "k", "w"]
        
        # envelopewの各成分の同時性の評価として、選ばれた２つの成分を掛け算したものを表示する
        if SHOW_PLOT:
            fig,ax = plt.subplots(NUM,1)
            
            # ax[0]に全体を表示する
            for i in range(NUM):
                ax[0].plot(x_time3, envelopew[i], color=color_list[i], label=F_LIST[i]) 
            
            ax[0].grid(which='both', axis='both')          
            ax[0].legend()
            if title is not None:
                ax[0].set_title( title )
            else:
                ax[0].set_title('Whole elements')
            
            # a[1],....に選ばれた２つの成分を掛け算したものを表示する
            for index_up in range(1,NUM):
                for i in range(NUM-index_up):
                    ax[index_up].plot(x_time3, envelopew[i] * envelopew[i+1], color=color_list[i], label=F_LIST[i+index_up]) 
                
                ax[index_up].grid(which='both', axis='both')
                ax[index_up].legend()
                ax[index_up].set_title(F_LIST[index_up-1])
            
            
            fig.tight_layout()
            plt.show()
            plt.close()
    
    def correlate2d(self, envelopew,  NUM=None,title=None, SHOW_PLOT=False):
        #
        if NUM is None:
            NUM=self.NUM_BPF
        color_list = ["r", "g", "b", "c", "m", "y", "k", "w"]
        
        #############################
        # データ数が多く、処理に時間がかかるので、ダウンサンプルする
        ##### Q_SKIP 50
        Q_SKIP=50 #13
        for i in range(len(envelopew)):
            if i == 0:
                test=signal.decimate(envelopew[i], Q_SKIP)
                template=signal.decimate(self.envelopew[i], Q_SKIP)
            else:
                test=np.vstack((test,signal.decimate(envelopew[i], Q_SKIP)))
                template=np.vstack((template,signal.decimate(self.envelopew[i], Q_SKIP)))
        if 0:
            print( 'Q_SKIP', Q_SKIP)
            print( 'test.shape', test.shape)
            print( 'template.shape', template.shape)
        
        
        corr=signal.correlate2d(test, template, boundary='fill', mode='valid')  #boundary='symm', 'valid')
        y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
        #print('corr.shape', corr.shape)
        #print('y, x', y, x)
        
        # 位置の計算
        x0= int(x * Q_SKIP)
        x0_time= x0 / self.sr_test_wav
        #print ('x0, x0_time ', x0, x0_time)
        
        #
        if SHOW_PLOT:
            fig = plt.figure()
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
            
            for i in range( corr.shape[0]):
                ax1.plot(corr[i,:], color=color_list[i]) 
            
            ax1.grid(which='both', axis='both')
            
            x_time3_test= np.linspace(0, len(test[0,:]), len(test[0,:]))
            x_time3_template= np.linspace(0, len(template[i,:]), len(template[i,:]))
            ax2.plot(x_time3_test, test[0,:], color=color_list[0]) 
            ax2.plot(x_time3_template+x, template[i,:], color=color_list[1]) 
            ax2.grid(which='both', axis='both')
            
            fig.tight_layout()
            plt.show()
            plt.close()
        
        return x0, x0_time
        
    
    def plot_wav2(self,test_wav,ref_wav,x0_time, title=None, SHOW_HPF=True):
        #
        x_time3_test= np.linspace(0, len(test_wav)/self.sr_chime, len(test_wav))
        x_time3_chime= np.linspace(0, len(ref_wav)/self.sr_chime, len(ref_wav))
        color_list = ["r", "g", "b", "c", "m", "y", "k", "w"]
        
        if SHOW_HPF:
            f_center=200 # set HPF fc
            hb, ha = signal.iirfilter(4, f_center ,  btype='highpass', ftype='butter', fs=self.sr_chime)
            hpf_out=signal.lfilter(hb, ha, test_wav)
        
        
        fig = plt.figure()
        
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        
        ax1.plot(x_time3_test, test_wav, color=color_list[0])
        ax1.plot(x_time3_chime + x0_time, ref_wav, color=color_list[1])
        #if SHOW_HPF:
        #    ax1.plot(x_time3_test, hpf_out, color=color_list[5])
        
        ax1.grid(which='both', axis='both')
        if title is not None:
            ax1.set_title(title)
        
        ax2.plot(x_time3_test, test_wav, color=color_list[0])
        ax2.plot(x_time3_chime + x0_time, ref_wav, color=color_list[1])
        if SHOW_HPF:
            ax2.plot(x_time3_test, hpf_out, color=color_list[5])
        
        ax2.grid(which='both', axis='both')
        ##############################################
        # 初期化時に　offsetx lng0 が手動設定となっている
        ax2.set_xlim(x0_time+self.offsetx, x0_time+self.offsetx+self.lng0)
        
        fig.tight_layout()
        
        plt.show()
        
        if 0 and (title is not None):
            save_path='figure/' + title + '.png'
            fig.savefig(save_path)
        
        plt.close()
        
        # 表示した部分の位置を返す
        #return int((x0_time+offsetx) * self.sr_chime) , int((x0_time+offsetx+lng0) * self.sr_chime)
    
    
    def plot_correlate(self, ref_signal, x0_time, another_signal, x0_time_an, title=None, save_path=None, SHOW_PLOT=True):
        #
        #  2つの信号の振幅のcorrelateでは、信号の先頭がDRCなどでつぶれている場合、誤推定する可能性がある。
        #
        s1= ref_signal[int((x0_time+self.offsetx) * self.sr_chime) : int((x0_time+self.offsetx+self.lng0) * self.sr_chime)]
        s2= another_signal[int((x0_time_an+self.offsetx) * self.sr_chime) : int((x0_time_an+self.offsetx+self.lng0) * self.sr_chime)]
        
        s1_org=self.test_wav[int((x0_time+self.offsetx) * self.sr_chime) : int((x0_time+self.offsetx+self.lng0) * self.sr_chime),self.ref_channel]
        s2_org=self.test_wav[int((x0_time_an+self.offsetx) * self.sr_chime) : int((x0_time_an+self.offsetx+self.lng0) * self.sr_chime),self.another_channel]
        
        #
        cor0= signal.correlate(s1, s2)
        diff0= np.argmax(cor0) - len(s1)
        # 符号がマイナスの場合
        if diff0 < 0:
            cor0inv=cor0[::-1]
            cor0half= cor0inv[len(s1):]
        else:
            cor0half= cor0[len(s1):]
        
        #　cor0halfの減衰の具合(時間)を調べる
        #  cor0halfの半分が1/4波長とし、それが減衰するLPFを掛ける
        fc=1.0/ ((len(cor0half)/2*4)/self.sr_chime)
        # print ('fc ', fc, 1/fc)
        lb, la = signal.iirfilter(2, fc ,  btype='lowpass', ftype='butter', fs=self.sr_chime)
        cor0half_en= np.abs(cor0half) # cor0halfの正値へ
        cor0half_lpf_out=signal.lfilter(lb, la, np.abs(cor0half_en)) # LPFを掛けて小刻みに変動する成分を除く
        
        x_time3_s1= np.linspace(0, len(s1)/self.sr_chime, len(s1)) + x0_time+self.offsetx
        x_time3_s2= np.linspace(0, len(s2)/self.sr_chime, len(s2)) + x0_time_an+self.offsetx
        color_list = ["r", "g", "b", "c", "m", "y", "k", "w"]
        # ref_channelを基準に、another_channelがどのようにずれるかを計算する。
        dt0= (x0_time_an - x0_time) - (diff0 / self.sr_chime)
        
        if SHOW_PLOT:
            fig = plt.figure()
            ax1 = fig.add_subplot(4, 1, 1)
            ax2 = fig.add_subplot(4, 1, 2)
            ax3 = fig.add_subplot(4, 1, 3)
            ax4 = fig.add_subplot(4, 1, 4)
            
            ax1.plot(x_time3_s1, s1, color='blue') 
            ax1.grid(which='both', axis='both')
            if title is not None:
                ax1.set_title(title)
            
            #
            ax2.plot(x_time3_s2, s2, color='red') 
            ax2.grid(which='both', axis='both')
            
            #
            x_time2= np.linspace(0, len(cor0half)/self.sr_chime, len(cor0half))
            ax3.plot(x_time2, cor0half ,color='blue' )
            # 符号がマイナスの場合もある
            ax3.plot( abs(diff0) / self.sr_chime, cor0half[ int(abs(diff0))], "x",color='blue')
            ax3.plot(x_time2, cor0half_en ,color='yellow' )
            ax3.plot(x_time2, cor0half_lpf_out ,color='m' )
            ax3.set_title( 'correlate ' + str(diff0 / self.sr_chime))
            ax3.grid(which='both', axis='both')
            
            #
            
            ax4.plot(x_time3_s1, s1, color='blue') 
            ax4.plot(x_time3_s2, s2, color='red') 
            # 符号がマイナスの場合
            if dt0 > 0:
                ax4.plot(x_time3_s1 + dt0, s1, color='yellow') 
            else:
                ax4.plot(x_time3_s2 - dt0, s2, color='yellow') 
            
            ####
            
            ax4.set_title('dt ' +  str(dt0))
            ax4.grid(which='both', axis='both')
            
            plt.tight_layout()
            plt.show()
            plt.close()
        
        
        # corrが速やかに減衰していないので、推定が上手く行っていないと判定する場合、エラーを返す。
        THRESHOLD_GNESUI=0.18 # 0.2 # 0.22 # 減衰していると判断する閾値
        if SHOW_PLOT:
            print ('ratio ',  cor0half_lpf_out[-1] / max( cor0half))
        if cor0half_lpf_out[-1] > max( cor0half) * THRESHOLD_GNESUI:
            plt.close()
            return False, dt0
        
        #
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax1.plot(x_time3_s1, s1_org, color='blue') 
        ax1.plot(x_time3_s2, s2_org, color='red') 
        ax1.grid(which='both', axis='both')
        if title is not None:
            ax1.set_title(title)
        
        
        # 符号がマイナスの場合
        if dt0 > 0:
            ax2.plot(x_time3_s1 + dt0, s1_org, color='yellow') 
            ax2.plot(x_time3_s2 , s2_org, color='red') 
        else:
            ax2.plot(x_time3_s2 - dt0 , s2_org, color='yellow') 
            ax2.plot(x_time3_s1, s1_org, color='blue') 
        
        ax2.set_title('estimation differential time ' +  str(dt0))
        ax2.grid(which='both', axis='both')
        
        plt.tight_layout()
        if SHOW_PLOT:
        	plt.show()
        
        # save as a figure
        if save_path is not None:
            fig.savefig(save_path)
        
        plt.close()
        
        return True, dt0 #  予想される L R の時間差を返す
    
    def main0(self, path_test_wav, title=None, acept_maximum_wav_length=15):
        # テスト信号にBPFをかける
        self.test_wav, self.sr_test_wav= self.read_wav( path_test_wav)
        print ('test wav is ', path_test_wav) #, self.sr_test_wav)
        
        save_path= self.save_dir + os.path.splitext(os.path.basename(path_test_wav))[0] + '.png'
        fname= self.save_dir + 'output.txt'
        
        # check
        if self.sr_test_wav != self.sr_chime:
            print ('error: sr is different.')
            self.write2text(path_test_wav, 'error: sr is different.', fname)
            return False, -101
        if len(self.test_wav.shape) == 1 or self.test_wav.shape[1] != 2:
            print ('error: test wav is not stereo')
            self.write2text(path_test_wav, 'error: test wav is not stereo.', fname)
            return False, -102
        if self.test_wav.shape[0] > acept_maximum_wav_length * self.sr_test_wav:
            print ('error: wav size  is more than aceptable maximum length.')
            self.write2text(path_test_wav, 'error: wav size  is more than aceptable maximum length.', fname)
            return False, -103
        
        # switch of what is shown
        SHOW_PLOT1=self.SHOW_PLOT
        SHOW_PLOT2=self.SHOW_PLOT2
        # BPFを掛けたものを合算したものを作る
        syth_out,_,test_envelopew= self.apply_bpf(self.test_wav[:,self.ref_channel],SHOW_PLOT=SHOW_PLOT1)
        
        self.show_douzisei(test_envelopew, title=path_test_wav, SHOW_PLOT=SHOW_PLOT1)
        
        
        # try to correlate2d
        x0, x0_time= self.correlate2d(test_envelopew, SHOW_PLOT=SHOW_PLOT2)
        path_test= os.path.splitext(os.path.basename(path_test_wav))[0]
        
        if SHOW_PLOT2:
            self.plot_wav2(self.test_wav[:,self.ref_channel], self.chime_wav[:,self.ref_channel], x0_time, path_test + '_ref')
        
        #
        if self.another_channel is not None:
            syth_out_an,_,test_envelopew_an= self.apply_bpf(self.test_wav[:,self.another_channel],SHOW_PLOT=self.SHOW_PLOT)
            self.show_douzisei(test_envelopew_an, title=path_test_wav+' another', SHOW_PLOT=SHOW_PLOT1)
            
            # try to correlate2d
            x0_an, x0_time_an= self.correlate2d(test_envelopew_an, SHOW_PLOT=SHOW_PLOT2)
            if SHOW_PLOT2:
                self.plot_wav2(self.test_wav[:,self.another_channel],self.chime_wav[:,self.ref_channel],x0_time_an, path_test + '_another')
            
            
            # WAVファイルとして書き出す
            if 0:
                file_path=os.path.join('syth_wav', os.path.splitext(os.path.basename(path_test_wav))[0] + '_sythout.wav')
                #                                  ↓順番に注意！
                self.save_wav(file_path, np.stack((syth_out_an, syth_out),-1)  , sr=self.sr_chime)
            
            
            # グラフのタイトル
            if title is not None:
                title_figure= title
            else:
                title_figure= os.path.basename(path_test_wav)
        	
            # ref_channel another_channelの時間差を推定する。
            rt_code_corr, dt0= self.plot_correlate(syth_out, x0_time, syth_out_an, x0_time_an, title=title_figure, save_path=save_path, SHOW_PLOT=SHOW_PLOT2)
            
            if not rt_code_corr :
                print('There is fault in estimation1 stage.')
                self.write2text(path_test_wav, 'There is fault in estimation1 stage.', fname)
                return False, -104
            
            print ('LR_estimation ', dt0)
            # write estimation result to text file
            self.write2text(path_test_wav, dt0, fname)
            
            return True, dt0
            
            
        return True, 0 # set dummy value as dt0
        
    
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


    def write2text(self,file_path, LR_estimation1_or_message, fname):
        ### write to text file
        if self.test_number == 0: # new
            self.write_text(file_path, LR_estimation1_or_message, fname=fname, mode='w')
        else:    # append 
            self.write_text(file_path, LR_estimation1_or_message, fname=fname, mode='a')
        #
        self.test_number=self.test_number+1  # count +1
    
    
    def write_text(self, file_path, value, fname='output.txt',mode='a'):
        with open( fname, mode, encoding='UTF-8') as f:
            dt_now = datetime.datetime.now()
            f.write(dt_now.isoformat())
            f.write(': ')
            f.write(file_path)
            f.write(',')
            if type(value) == str:
                f.write(value)
            else:
                f.write(str(value))
            f.write('\n')
            f.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='estimate L and  R signal difference time')
    parser.add_argument('--ref_channel', '-r', type=int, default=1, help='specify ref channel of L(0) R(1)')
    parser.add_argument('--dir', '-d', default='WAV/', help='specify test wav directory')
    parser.add_argument('--another_channel', '-t', type=int, default=0, help='specify another channel L(0) R(1)')
    parser.add_argument('--clime_wav', '-c', default='sample_wav/chime_only.wav', help='specify chime wav as reference')
    args = parser.parse_args()
    
    
    # trial some test wav files
    if 0:
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
    tde1= time_difference_estimation(args.clime_wav, args.ref_channel, args.another_channel, SHOW_PLOT=SHOW, ShowOntheWay=SHOW, SHOW_PLOT2=SHOW)
    
    t_time_list=[]
    # 試験用のwavを読ませて時間差を推定してみる
    for i,file_path in enumerate(flist):
        rtcode,t_time= tde1.main0( file_path, acept_maximum_wav_length=20)
        if rtcode:
            t_time_list.append(t_time)
    
    # 集計結果を表示する
    print('number of count', len(t_time_list))
    print('max ', max(t_time_list))
    print('min ', min(t_time_list))