import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
from dataset import MyDataset
import numpy as np
import time
from model import LipNet
import torch.optim as optim
import re
import json
import tempfile
import shutil
import cv2
import face_alignment

FILE_PATH = 'data/frames/i_v/i_v0/i_v0_0_frame'
FILE_OUT = 'samples/frames'

def get_position(size, padding=0.25):

    x = [0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
                    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
                    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
                    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
                    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
                    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
                    0.553364, 0.490127, 0.42689]

    y = [0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
                    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
                    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
                    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
                    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
                    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
                    0.784792, 0.824182, 0.831803, 0.824182]

    x, y = np.array(x), np.array(y)

    x = (x + padding) / (2 * padding + 1)
    y = (y + padding) / (2 * padding + 1)
    x = x * size
    y = y * size
    # xとyを1組のセットにしてその組み合わせを配列にして、numpy配列に変換している
    return np.array(list(zip(x, y)))

def cal_area(anno):
    return (anno[:,0].max() - anno[:,0].min()) * (anno[:,1].max() - anno[:,1].min())

# 処理後のビデオの出力
def output_video(p, txt, dst):
    files = os.listdir(FILE_PATH)
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    # ビデオへの文字列書き込み処理
    font = cv2.FONT_HERSHEY_SIMPLEX

    for file, line in zip(files, txt):
        img = cv2.imread(os.path.join(FILE_PATH, file))
        h, w, _ = img.shape
        img = cv2.putText(img, line, (w//8, 11*h//12), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        img = cv2.putText(img, line, (w//8, 11*h//12), font, 0.5, (255, 255, 255), 0, cv2.LINE_AA)
        #h = h // 2
        #w = w // 2
        img = cv2.resize(img, (w, h))
        cv2.imwrite(os.path.join(p, file), img)
    # 処理後のフレームからビデオを作成するコマンド(1s毎に25枚の画像を生成)
    cmd = "ffmpeg -y -i {}/%02d.jpg -r 30 \'samples/{}\'".format(FILE_OUT, dst)
    os.system(cmd)

# TODO: 正規化や特異値分解で次元の削減
def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    # 列ごとの平均値を返す
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)

    # 全ての要素に-平均値をする
    points1 -= c1
    points2 -= c2
    # 標準偏差を求める
    s1 = np.std(points1)
    s2 = np.std(points2)
    # 正規化
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2) #points1の転置とpoints2の積の値の特異値分解
    R = (U * Vt).T # UとVtの積を転置
    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])

# ビデオの読み込み
def load_video():
    #p = tempfile.mkdtemp() # 一時的にメモリにディレクトリを作成 (電源を落とすと削除される)
    # サンプリングしてフレームを作成
    #cmd = 'ffmpeg -i \'{}\' -qscale:v 2 -r 25 \'{}/%d.jpg\''.format(file, p)
    #cmd = '{}'.format(file)
    #sys.path.append('..')
    #os.system(cmd)

    files = os.listdir(FILE_PATH) # ディレクトリの中身の一覧を配列で返す
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    # os.path.splitext() : 拡張子の取得
    # lambdaは無名関数


    # 一時ディレクトリに保存されているフレームの画像ファイルを順にロードして、配列として取得する
    array = [cv2.imread(os.path.join(FILE_PATH, file)) for file in files]

    # データがないものをフィルタ処理をしたデータの配列を返す
    array = list(filter(lambda im: not im is None, array))
    #array = [cv2.resize(im, (100, 50), interpolation=cv2.INTER_LANCZOS4) for im in array]

    # 2Dの画像の人の顔のランドマークを検出するオブジェクトのインスタンスを生成
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

    #fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda') GPU使用の場合これを使う
    # フレームの画像のランドマークを配列として取得
    points = [fa.get_landmarks(I) for I in array]


    front256 = get_position(256)
    video = []

    for point, scene in zip(points, array): # フレームのランドマークと対応しているフレームの画像を1セットにしてそれを順に回す。
    #(point: ランドマーク, scene: ランドマークに対応している画像)
        if(point is not None): # ランドマークがある場合
            shape = np.array(point[0])
            # 口元部分のランドマークだけをとる
            shape = shape[17:]
            M = transformation_from_points(np.matrix(shape), np.matrix(front256))

            img = cv2.warpAffine(scene, M[:2], (256, 256)) # アフィン変換 計算コストの軽量化
            (x, y) = front256[-20:].mean(0).astype(np.int32)
            w = 160//2
            img = img[y-w//2:y+w//2,x-w:x+w,...]
            img = cv2.resize(img, (128, 64)) # 128x64の大きさに変換
            video.append(img) # 処理後の画像のデータを追加


    video = np.stack(video, axis=0).astype(np.float32) # float32のnumpy配列にする
    video = torch.FloatTensor(video.transpose(3, 0, 1, 2)) / 255.0 # テンソルのに変換

    return video# 処理後のビデオと一時的に作ったディレクトリの情報を返す

# ctc
def ctc_decode(y):
    y = y.argmax(-1) # 横軸での最大要素を取り出す
    t = y.size(0) # 最大要素のサイズを代入
    print(t)
    exit()

    result = []
    for i in range(t+1):
        result.append(MyDataset.ctc_arr2txt(y[:i], start=1)) # ctcの配列要素を文字に変換して結果に追加
    return result


if(__name__ == '__main__'):
    # optionファイルに設定している変数などをロード
    opt = __import__('options')

    # cpu か gpu のどちらを使うか指定
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    #  LipNetの定義
    model = LipNet()
    # cuda() : gpuへの切り替え
    model = model.cuda()
    net = nn.DataParallel(model).cuda()
    #net = nn.DataParallel(model) # データの並列処理化

    if(hasattr(opt, 'weights')): # optionsからロードしたオブジェクトに 'weights' があるか判定
        pretrained_dict = torch.load(opt.weights) # 学習済み重みをロード
        model_dict = model.state_dict() # 重みパラメータの容量の削減 (また、state_dictはいらない次元の削減できるため軽量に保存できる)

        # 条件に合うのもだけをpretrained_dictに保存する
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        # pretrained_dictに入っていない場合それを保存する
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        # ロードできたパラメータの表示
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        # パラメータミスの表示
        print('miss matched params:{}'.format(missed_params))
        # モデルの重みをフィルタリングした学習済みの重みに変換
        model_dict.update(pretrained_dict)
        # モデルに上記の重みをロード
        model.load_state_dict(model_dict)

    # 軽量処理を行ったビデオデータとフレーム情報を取得
    video = load_video() # コマンドラインの python demo.py argv//ここを取得
    y = model(video[None,...].cuda()) # gpuへ切り替え
    #y = model(video[None,...]) # ビデオの情報をモデルへ
    txt = ctc_decode(y[0]) # 推測した結果をテキスト化

    output_video(FILE_OUT, txt, sys.argv[1]) # 結果をフレームに書いてビデオに
