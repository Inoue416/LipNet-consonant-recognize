import numpy as np
import cv2
import face_alignment


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

def get_lips(array):
    # 2Dの画像の人の顔のランドマークを検出するオブジェクトのインスタンスを生成
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

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

    #video = np.stack(video, axis=0).astype(np.float32) # float32のnumpy配列にする
    #video = torch.FloatTensor(video.transpose(3, 0, 1, 2)) / 255.0 # テンソルのに変換

    return video# return lip frames
