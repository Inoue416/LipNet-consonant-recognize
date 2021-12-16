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
from tensorboardX import SummaryWriter
import options as opt



if(__name__ == '__main__'):
    opt = __import__('options')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    writer = SummaryWriter(log_dir=opt.log_dir) # ダッシュボード作成

# データセットをDataLoaderへ入れてDataLoaderの設定をして返す
def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=True):
    return DataLoader(dataset,
        batch_size = opt.batch_size,
        shuffle = shuffle,
        num_workers = num_workers, # マルチタスク
        drop_last = True) # Trueにすることでミニバッチから漏れた仲間外れを除去できる (Trueを検討している)

# 学習率を返す
def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()

# CTC損出関数での結果を文字へと変換する
def ctc_decode(y):
    result = []
    y = y.argmax(-1) 
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]

# テスト
def test(model, net):

    with torch.no_grad(): # テストなどで勾配は求めない処理
        # テストデータのロード
        dataset = MyDataset(opt.video_path,
            opt.anno_path,
            opt.val_list,
            opt.vid_padding,
            opt.txt_padding,
            'test')

        #print('num_test_data:{}'.format(len(dataset.data)))
        model.eval() # テストモードへ
        loader = dataset2dataloader(dataset, shuffle=False) # DataLoaderを作成
        loss_list = []
        wer = []
        cer = []
        crit = nn.CTCLoss()
        tic = time.time()
        for (i_iter, input) in enumerate(loader):
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
            #print(vid_len)
            #print(txt_len)
            y = net(vid) # ネットへビデオデータを入れる
            # 損出関数での処理
            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy()
            #print(loss)
            # 損出関数の値を記録
            loss_list.append(loss)
            # 結果の文字を入れる
            pred_txt = ctc_decode(y)
            # 正しい文字列を入れる
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            # 単語及び、文字のエラー率を算出
            wer.extend(MyDataset.wer(pred_txt, truth_txt))
            cer.extend(MyDataset.cer(pred_txt, truth_txt))

            # 条件の回数の時だけエラー率などを表示
            if(i_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(i_iter+1)
                eta = v * (len(loader)-i_iter) / 3600.0

                print(''.join(101*'-'))
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-'))
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:4]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101 *'-'))
                print('test_iter={},eta={},wer={},cer={}'.format(i_iter,eta,np.array(wer).mean(),np.array(cer).mean()))
                print(''.join(101 *'-'))

        return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())

# 訓練
def train(model, net):

    # データのロード
    dataset = MyDataset(opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train')

    # DataLoaderの作成
    loader = dataset2dataloader(dataset)
    # optimizerの初期化(Adam使用)
    optimizer = optim.Adam(model.parameters(),
                lr = opt.base_lr,
                weight_decay = 0.1,#0.1, # パラメータのL2ノルムを正規化としてどれくらい用いるから指定
                amsgrad = True)# AMSgradを使用する

    #print('num_train_data:{}'.format(len(dataset.data)))
    crit = nn.CTCLoss()
    tic = time.time()
    train_wer = []
    train_cer = []
    loss_list = []
    for epoch in range(opt.max_epoch): # epoch分学習する
        for (i_iter, input) in enumerate(loader):
            model.train() # 訓練モードへ
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()

            #if vid_len.view(-1) < txt_len.view(-1):
            #print('vl : {}'.format(vid_len.view(-1)))
            #print('tl : {}'.format(txt_len.view(-1)))

            optimizer.zero_grad() # パラメータ更新が終わった後の勾配のクリアを行っている。
            y = net(vid) # ビデオデータをネットに投げる
            # 損出を求める
            #exit()
            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
            loss_list.append(loss)
            #print(loss)
            # 損出をもとにバックワードで学習
            loss.backward()

            if(opt.is_optimize):
                optimizer.step() # gradプロパティに学習率をかけてパラメータを更新

            tot_iter = i_iter + epoch*len(loader) # 現在のイテレーション数の更新

            pred_txt = ctc_decode(y) # 結果を文字へ
            
            
            # 正解の文字列をロード
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
        
            #exit()
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt)) # エラー率を算出
            train_cer.extend(MyDataset.cer(pred_txt, truth_txt))

            # 条件の回数の時、それぞれの経過を表示
            if(tot_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (len(loader)-i_iter)*v/3600.0

                writer.add_scalar('train loss', loss, tot_iter)
                writer.add_scalar('train wer', np.array(train_wer).mean(), tot_iter)
                print(''.join(101*'-'))
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-'))

                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101*'-'))
                print('epoch={},tot_iter={},base_lr={},eta={},loss_mean={},loss={},train_wer={},train_cer={}'.format(epoch, tot_iter, opt.base_lr, eta, torch.mean(torch.stack(loss_list)), loss, np.array(train_wer).mean(), np.array(train_cer).mean()))
                print(''.join(101*'-'))
                #savename = 'base_lr{}_{}_loss_{}_wer_{}_cer_{}.pt'.format(opt.base_lr, opt.save_prefix, torch.mean(torch.stack(loss_list)),  np.array(train_wer).mean(), np.array(train_cer).mean()) # 学習した重みを保存するための名前を作成
                #(path, name) = os.path.split(savename)
                #if(not os.path.exists(path)): os.makedirs(path) # 重みを保存するフォルダを作成する
                #torch.save(model.state_dict(), savename) # 学習した重みを保存
                #if(not opt.is_optimize):
                    #exit()




            if(tot_iter % opt.test_step == 0):
                (loss, wer, cer) = test(model, net)
                print('i_iter={},lr={},loss={},wer={},cer={}'
                    .format(tot_iter,show_lr(optimizer),loss,wer,cer))
                writer.add_scalar('val loss', loss, tot_iter)
                writer.add_scalar('wer', wer, tot_iter)
                writer.add_scalar('cer', cer, tot_iter)
                savename = 'base_lr{}_{}_loss_{}_wer_{}_cer_{}.pt'.format(opt.base_lr, opt.save_prefix, loss,  wer, cer) # 学習した重みを保存するための名前を作成
                (path, name) = os.path.split(savename)

                #savename = '{}_loss_{}_wer_{}_cer_{}.pt'.format(opt.save_prefix, loss, wer, cer) # 学習した重みを保存するための名前を作成
                #(path, name) = os.path.split(savename)
                if(not os.path.exists(path)): os.makedirs(path) # 重みを保存するフォルダを作成する
                torch.save(model.state_dict(), savename) # 学習した重みを保存
                if(not opt.is_optimize):
                    exit()

if(__name__ == '__main__'):
    print("Loading options...")
    model = LipNet() # モデルの定義
    model = model.cuda() # gpu使用
    net = nn.DataParallel(model).cuda() # データの並列処理化

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(f=opt.weights)# 学習済みの重みをロード
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    # ネットワークの挙動に再現性を持たせるために、シードを固定して重みの初期値を固定できる
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    # 訓練開始
    train(model, net)
