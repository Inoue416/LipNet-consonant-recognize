# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import glob
import re
import copy
import json
import random
import editdistance
#from extract_lips import get_lips



class MyDataset(Dataset):
    letters = [' ', 'あ', 'か', 'さ', 'た', 'な', 'は', 'ま', 'や', 'ら', 'わ']


    def __init__(self, video_path, anno_path, file_list, vid_pad, txt_pad, phase):
        self.anno_path = anno_path # 正解文字列データのパスを格納
        self.vid_pad = vid_pad # ビデオデータのパッディングの数
        self.txt_pad = txt_pad # テキストのパッディング
        self.phase = phase # train か testのフェーズを格納

        # ビデオまでのパスを読み出す
        with open(os.path.join(video_path, file_list), 'r') as f:
            self.videos = [os.path.join(video_path, line.rstrip()) for line in f.readlines()]

        # データを作成
        self.data = []
        for vid in self.videos:
            items = vid.split(os.path.sep)
            self.data.append((vid, os.path.join(items[2], items[3])))
            #self.data.append((vid, os.path.join(items[0], items[1], items[2]), os.path.join(items[2], items[3]), items[-1]))
        self.video_path = video_path
    # データのロード
    def __getitem__(self, idx):
        (vid, anno_name) = self.data[idx] # spkビデオの入っているフォルダまでのパス
        #(vid, spk, anno_name, name) = self.data[idx] # spkビデオの入っているフォルダまでのパス
        #vid = self._make_frame(spk, vid) # フレームの作成
        vid = self._load_vid(vid) # ビデオデータのフレームをロード
        # 修正箇所
        anno = self._load_anno(os.path.join(self.anno_path, anno_name+'.txt')) # 正解の文字列データのロード

        # trainの場合、水平(垂直)反転
        if(self.phase == 'train'):
            vid = HorizontalFlip(vid)

        # 色の標準化
        vid = ColorNormalize(vid)

        vid_len = vid.shape[0] # ビデオの数
        anno_len = anno.shape[0] # アノテーションの数
        vid = self._padding(vid, self.vid_pad) # ビデオのパッディング
        anno = self._padding(anno, self.txt_pad) # アノテーションのパッディング

        return {'vid': torch.FloatTensor(vid.transpose(3, 0, 1, 2)),
            'txt': torch.LongTensor(anno),
            'txt_len': anno_len,
            'vid_len': vid_len}

    # データの長さを返す関数
    def __len__(self):
        return len(self.data)
    


    # ビデオのロード
    def _load_vid(self, p):
        files = os.listdir(p) # パスで指定された場所に入っているものの一覧を配列にして返す
        # フィルタリング .jpgを見つける
        files = list(filter(lambda file: file.find('.jpg') != -1, files))# 構文みたいなもの
        files = sorted(files, key=lambda file: int(os.path.splitext(file)[0])) # splitextは拡張子(.jpgなど)を抽出できる
        array = [cv2.imread(os.path.join(p, file)) for file in files] # フレームデータのロード
        array = list(filter(lambda im: not im is None, array)) # データないものをフィルタリングする
        #array = get_lips(array)
        #array = [cv2.resize(im, (128, 64), interpolation=cv2.INTER_LANCZOS4) for im in array]
        array = np.stack(array, axis=0).astype(np.float32)
        return array

    # アノテーションのロード
    def _load_anno(self, name):
        with open(os.path.join(self.video_path, name), 'r') as f:
            """lines = [re.sub('\n', '', line) for line in f.readlines()]
            txt = lines[0]"""
            lines = (f.read()).split('\n')
            lines.remove('')
            txt = lines[0].split(' ')
            text = []
            for t in txt:
                text.append(t.upper())
        return MyDataset.txt2arr(' '.join(text), 1)
        #return MyDataset.txt2arr(' '.join(txt.upper()), 1)
        #return MyDataset.txt2arr(txt.upper(), 1)

    # パッディング
    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)

    """def _make_frame(self, save_path, vid):
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            print('Not Found.')
            exit()
        save_ = os.path.splitext(os.path.basename(vid))[0]
        if not os.path.isdir(os.path.join(save_path, save_+'_frame')):
            os.makedirs(os.path.join(save_path, save_+'_frame'))
        re_path = os.path.join(save_path, save_+'_frame')
        ext='.jpg'
        digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        n = 0
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(('{}/{}{}').format(re_path, str(n).zfill(digit), ext), frame)
                n += 1
            else:
                return re_path"""

    @staticmethod
    def txt2arr(txt, start):
        arr = []
        for c in list(txt):
            arr.append(MyDataset.letters.index(c) + start)
        return np.array(arr)

    @staticmethod
    def arr2txt(arr, start):
        txt = []
        for n in arr:
            if(n >= start):
                txt.append(MyDataset.letters[n - start])
        return ''.join(txt).strip()

    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):
                if(len(txt) > 0 and txt[-1] == ' ' and MyDataset.letters[n - start] == ' '):
                    pass
                else:
                    txt.append(MyDataset.letters[n - start])
            pre = n
        return ''.join(txt).strip()

    @staticmethod
    def wer(predict, truth):
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return wer

    @staticmethod
    def cer(predict, truth):
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return cer