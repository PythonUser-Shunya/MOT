import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

df = pd.read_csv(r"csv\Term1_night_オサムシのすべて.csv", header=0)
id_max = df["ID"].max()

df_list = []
for i in range(1, id_max + 1):
    df_list.append(df[df["ID"] == i])

img = cv2.imread(r"image\Term1_night_0_1.jpg")
width = int(img.shape[0])
height = int(img.shape[1])
# 真っ白画像
img = np.ones(shape=(width, height, 3), dtype=np.int32) * 255

# 六角形の座標
point = [(489, 3), (290, 312), (480, 696), (920, 713), (1145, 343), (942, 1)]
pts = np.array(point)

# ホールの座標
fall_center = [(581, 88), (430, 320), (557, 570),
               (848, 585), (991, 341), (863, 93)]

# 5, 4, 3, 2, 1, 6
fall_color = [(0, 0, 255), (0, 255, 0), (255, 241, 0),
              (231, 121, 40), (255, 0, 0), (155, 114, 176)]
# フォール描画
for i, j in enumerate(fall_center):
    img = cv2.circle(img, j, 15, fall_color[i], -1)

# 六角形描画
cv2.polylines(img, [pts], True, (0, 0, 0), thickness=1)

# text_position_y = 50
# id = 1

# 軌跡を描画
color = (0, 0, 0)
for i in df_list:
    x = i["X"]
    y = i["Y"]
    center = [(int(x), int(y)) for x, y in zip(x, y)]
    df_center = [(int(x), int(y)) for x, y in zip(x, y)]
    pts = np.array(df_center)
    # 線を引く場合
    cv2.polylines(img, [pts], False, color, thickness=2)
    # 矢印の場合。矢印の間隔を設定するためにenumerateを使用
    # prev_center = df_center[0]
    # for number, center in enumerate(df_center):
    # if number % 40 == 0:
    # cv2.arrowedLine(img, prev_center, center,
    #         color, thickness=2, tipLength=0.2)
    # prev_center = center
    cv2.putText(img, f"Total: {id_max}", (1100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# plt.grid()
plt.imshow(img)
plt.savefig("image/osamushi.jpg")
