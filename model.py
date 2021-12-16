import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np


class LipNetConsonant(torch.nn.Module):
    def __init__(self, dropout_p=0.5):
        super(LipNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        #self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        # TODO: video_len_max is hiden state size. vlm equal video_len_max
        self.gru1  = nn.GRU(96*4*8, video_len_max, 1, bidirectional=True)
        self.gru2  = nn.GRU(vlm*2, vlm, 1, bidirectional=True)

        self.FC    = nn.Linear(vlm*2, 10+1)

        #self.gru1  = nn.GRU(96*4*8, 256, 1, bidirectional=True)
        #self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)

        #self.FC    = nn.Linear(512, 27+1)
        self.dropout_p  = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)
        self._init()

    def _init(self):

        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu') # He初期化 重みの初期化
        init.constant_(self.conv1.bias, 0) # 入力テンソルに値を入れる constant_(テンソル, 埋める値) バイアスの初期化

        #以下上記と同じ
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.constant_(self.conv2.bias, 0)

        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        init.constant_(self.conv3.bias, 0)

        init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
        init.constant_(self.FC.bias, 0)

        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + vlm))
            #stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, vlm * 3, vlm):
                init.uniform_(m.weight_ih_l0[i: i + vlm],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv) # 入力されたテンソルを一様分布から引き出された値で埋める
                #init.uniform_(m.weight_ih_l0[i: i + 256],
                            #-math.sqrt(3) * stdv, math.sqrt(3) * stdv) # 入力されたテンソルを一様分布から引き出された値で埋める

                # 入力テンソルを(半)直交行列で埋める
                # また、入力するテンソルは少なくとも2次元は必要であり、2次元を超える場合、後続の次元は平坦化される。
                # 重みの初期化

                init.orthogonal_(m.weight_hh_l0[i: i + vlm])
                # バイアスの初期化
                init.constant_(m.bias_ih_l0[i: i + vlm], 0)

                # 上記と同じ
                init.uniform_(m.weight_ih_l0_reverse[i: i + vlm],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + vlm])
                init.constant_(m.bias_ih_l0_reverse[i: i + vlm], 0)

                #init.orthogonal_(m.weight_hh_l0[i: i + 256])
                # バイアスの初期化
                #init.constant_(m.bias_ih_l0[i: i + 256], 0)

                # 上記と同じ
                #init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                            #-math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                #init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                #init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool3(x)

        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous() # 軸の順番を変更
        # また、contiguous()はviewにするとき、メモリ上に要素順に並ぶため、エラーを回避できる
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1).contiguous()

        # RNNの重みがメモリ上で非連続にならないように
        # また、パラメータデータポインターをリセットして、より高速なコードパスを使用できるようになっている。
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()

        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x)

        x = self.FC(x)
        x = x.permute(1, 0, 2).contiguous()
        return x
