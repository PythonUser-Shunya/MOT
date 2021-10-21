from scipy.spatial import distance as dist
import numpy as np
import random
import cv2

MAX_DISTANCE = 20

class CentroidTracker:
    def __init__(self, maxDisappeared=100):
        # objectは番号で管理( += 1)するから0で初期化
        self.nextObjectID = 0
        # object更新用{object番号:  中心点}
        self.objects = {}
        # 軌跡描画用{object番号:  中心点のリスト}
        self.center_list = {}
        # 虫によって色を変えるため辞書型に。{object番号:  色}
        self.color_dict = {}
        # 消滅させる用。消滅ポイント的な
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared

    # register: 登録
    # 作成するもの：座標を保持するlist
    def register(self, centroid):
        # 座標情報と色情報は保持したいからlistにする
        self.center_list[self.nextObjectID] = []
        # オブジェクトを登録するとき、次に使用可能なオブジェクトIDを使用して重心を格納
        self.objects[self.nextObjectID] = centroid
        # 合計N個の後続フレームで既存のオブジェクトと一致できない場合、古いオブジェクトの登録を解除
        # += 1でフレーム数をカウントするために0にしておく。
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    # deredister: 登録解除
    def deregister(self, objectID):
        # オブジェクトIDの登録を解除するには、それぞれの辞書の両方からオブジェクトIDを削除
        # del self.objects[objectID]
        # del self.disappeared[objectID]
        pass

    def draw(self, objectID, centroid, frame):
        # 中心点を追加
        self.center_list[objectID].append(centroid)
        # 色を追加
        self.color_dict.setdefault(objectID, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        text = f"ID:{objectID}"
        # 各objectIDに対応する中心点リストの最後の配列をもとにテキストを表示させる
        cv2.putText(frame, text, (self.center_list[objectID][-1][0],
                    self.center_list[objectID][-1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_dict[objectID], 1)
        # 過去の座標(スタート座標)を格納
        prev_center = self.center_list[objectID][0]
        for center in self.center_list[objectID]:
            cv2.line(frame, tuple(prev_center), tuple(center),
                        self.color_dict[objectID], thickness=1)
            # ゴールはスタートになる 
            prev_center = center


    def update(self, rects):
        if len(rects) == 0:
            # 矩形がないときはこれまでのobjectが消えた可能性があるから
            # 現時点で保持しているobject（各虫たち）の消滅ポイントを上げる
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # 最大長滅ポイントに達したらそのobjectを消滅させる
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            # オブジェクトを消す場合にはもうほかに作業が必要ないからreturn
            return self.objects
        
        # 現在のフレームの入力重心の配列を初期化
        # 矩形の数だけ0行列を作る。この行列に距離情報を格納
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # バウンディングボックスの長方形をループ
        # 矩形の数だけ中心点が計算される
        for (i, (x, y, w, h)) in enumerate(rects):
            # バウンディングボックスの座標を使用して図心を導出
            cX = x + w // 2.0
            cY = y + h // 2.0
            inputCentroids[i] = (cX, cY)

        # 現在オブジェクトを追跡していない場合は、入力重心を取得してそれぞれを登録
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        # それ以外の場合は、現在オブジェクトを追跡しているため、
        # 入力重心を既存のオブジェクト重心と一致させる必要がある。
        else:
            # オブジェクトIDと対応する中心点を取得
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # オブジェクト重心と入力重心の各ペア間の距離をそれぞれ計算。
            # 目標は、入力重心を既存のオブジェクト重心に一致させること。
            D = dist.cdist(np.array(objectCentroids),
                           inputCentroids, metric='euclidean')
        
            # このマッチングを実行するには、
            # （1）各行で最小値を見つけ、
            # （2）最小値に基づいて行インデックスを並べ替えて、
            # 最小値の行がインデックスリストの*最前線*になるようにする必要がある。
            rows = D.min(axis=1).argsort()
            # 各列で最小値を見つけ、以前に計算された行インデックスリストを使用して
            # 並べ替えることにより、列に対して同様のプロセスを実行。
            cols = D.argmin(axis=1)[rows]
            # inオブジェクトを更新、登録、または登録解除する必要があるかどうかを判断するには、
            # すでに調べた行と列のインデックスを追跡する必要がある。
            usedRows = set()
            usedCols = set()
            # （行、列）インデックスタプルの組み合わせをループ
            for (row, col) in zip(rows, cols):
                # 以前に行または列の値のいずれかをすでに調べたことがある場合は、それを無視
                if row in usedRows or col in usedCols:
                    continue
                # それ以外の場合は、現在の行のオブジェクトIDを取得し、
                # 新しい重心を設定して、消えたカウンターをリセット。
                for d in D:
                    if d[0] <= MAX_DISTANCE:
                        objectID = objectIDs[row]
                        self.objects[objectID] = inputCentroids[col]
                # 行インデックスと列インデックスをそれぞれ調べたことを示す
                usedRows.add(row)
                usedCols.add(col)
            # まだ調べていない列のインデックスを計算
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            # オブジェクトの重心の数が入力の重心の数以上である場合は、
            # これらのオブジェクトの一部が潜在的に消えているかどうかを確認する必要があります。
            # if D.shape[0] >= D.shape[1]:
                # 未使用の行インデックスをループします
                # for row in unusedRows:
                    # 対応する行インデックスのオブジェクトIDを取得し、
                    # 消えたカウンターをインクリメントします
                    # objectID = objectIDs[row]
                    # self.disappeared[objectID] += 1
                    # オブジェクトの登録を解除するために、
                    # オブジェクトが「消えた」とマークされた連続フレームの数を確認します
                    # if self.disappeared[objectID] > self.maxDisappeared:
                    #     self.deregister(objectID)
            # else:
            for col in unusedCols:
                self.register(inputCentroids[col])
                # 追跡可能なオブジェクトのセットを返します
        return self.objects
