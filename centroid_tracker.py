import cv2
from Class_centroid_tracker2 import CentroidTracker

# cap = cv2.VideoCapture(r"D:\00003_Trim.mp4")
# cap = cv2.VideoCapture(r"C:\openCv\ypppi-\movie\track.mp4")
cap = cv2.VideoCapture(r"C:\openCv\ypppi-\movie\multi.mp4")
# cap = cv2.VideoCapture(r"C:\openCv\ypppi-\black_00003_Trim.avi")
fgbg = cv2.createBackgroundSubtractorMOG2()
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # カメラ画像の横幅を1280に設定
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# VideoWriter を作成する。
# fourcc = cv2.VideoWriter_fourcc(*"DIVX")
# 動画の仕様（ファイル名、fourcc, FPS, サイズ, カラー）
# writer = cv2.VideoWriter("tracking2.avi", fourcc, fps, (width, height))
ct = CentroidTracker()

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    
    rects = []
    # フィルタでぼかして二値かして虫を大きくする。
    frame_blur = cv2.GaussianBlur(frame, (15, 15), 0)
    fgmask = fgbg.apply(frame_blur)
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_over40 = list(filter(lambda x: cv2.contourArea(x) >= 40 and cv2.contourArea(x) <= 150, contours))
    # retval, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)
    for cnt in contours_over40:
        xywh = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (xywh[0], xywh[1]), (xywh[0]+xywh[2], xywh[1]+xywh[3]), (255, 0, 0), 1)
        rects.append(xywh)

    if rects :
        objects = ct.update(rects)
        for (objectID, centroid) in objects.items():
            # オブジェクトのIDとオブジェクトの図心の両方を出力フレームに描画します
            ct.draw(objectID, centroid, frame)
        frame = cv2.resize(frame, dsize=(960, 540))
        fgmask = cv2.resize(fgmask, dsize=(960, 540))
        cv2.imshow("Frame", frame)
        # cv2.imshow("fgmask", fgmask)
        # writer.write(frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()
