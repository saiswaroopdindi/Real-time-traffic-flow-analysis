import cv2
from PIL import Image, ImageChops
import numpy as np
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ', '+ str(y)
        cv2.putText(img, strXY, (x, y), font, .5, (255, 255, 0), 2)
        cv2.imshow('image', img)
cap = cv2.VideoCapture("samplevideo.mp4")
ri= cv2.imread('back.png')
mw=2560 #my computer screen width
ri= cv2.resize(ri, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
ret, frame = cap.read()
cv2.imwrite('back1.png', frame)
ri=cv2.imread('back1.png')
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
fw= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))#fw=frame width
fh= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))#fh=frame height
wf = np.ones((fh, fw, 3), dtype=np.uint8) * 255 #wf=white frame
ra=884766#ra=road area
acc= np.zeros((fh, fw, 3), dtype=np.float32)#acc=accumulator
fr= cap.get(cv2.CAP_PROP_FPS)#fr=frame rate
f=50*fr#f=no.of frames or for no.of seconds
fc= 0#fc=frame count
while f>=0:
    ret, frame = cap.read()
    if not ret:
        break
    acc+= frame
    fc+= 1
    f-=1
le= acc/ fc
le= cv2.convertScaleAbs(le)#le=long exposure image
img = cv2.imread('le.png')
tps=0#tps=traffic in previous second/frame
wi = np.ones((1000, 2000, 3), dtype=np.uint8) * 0 #wi=white image
while True:
    ret, f2 = cap.read()
    ret, f3 = cap.read()
    frame1=f2
    frame=f2
    if not ret:
        print("Error: Could not read frame from video stream.")
        break
    difference = cv2.absdiff(frame, le)
    mask = object_detector.apply(frame1)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    a1=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            # cv2.drawContours(frame1, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            a1+=(w*h)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 1)
    gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, binary_difference = cv2.threshold(gray_difference, 30, 255, cv2.THRESH_BINARY)
    for i in range(0,300,10):
        binary_difference=cv2.line(binary_difference,(0,i),(fw,i),(0,0,0),10)
        wf = cv2.line(wf, (0, i), (fw, i), (0, 0, 0), 10)
    for i in range(300,570,5):
        binary_difference=cv2.line(binary_difference,(0,i),(540,300),(0,0,0),10)
        wf = cv2.line(wf, (0, i), (540, 300), (0, 0, 0), 10)
    for i in range(300,674,10):
        binary_difference=cv2.line(binary_difference,(765,300),(fw,i),(0,0,0),10)
        wf= cv2.line(wf, (765, 300), (fw, i), (0, 0, 0), 10)
    # for filling quadrilateral
    points = np.array([[631, 300], [676, 300], [480, fh], [763, fh]], np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.fillPoly(binary_difference, [points], (0, 0, 0))
    cv2.fillPoly(wf, [points], (0, 0, 0))
    cv2.fillPoly(binary_difference, [points], (0, 0, 0))
    points = np.array([[631, 300], [763, fh], [676, 300], [480, fh]], np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.fillPoly(binary_difference, [points], (0, 0, 0))
    cv2.fillPoly(wf, [points], (0, 0, 0))
    #for removing noise / samll white pixels
    _, binary_image = cv2.threshold(binary_difference, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary_difference= cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    ta=np.sum(binary_difference == 255) #ta=traffic area
    tp=2*(ta*100)/ra #tp=traffic percentage
    ti=ta-tps #traffic increased
    tps=ta
    blk = np.zeros((300, 800, 3), dtype=np.uint8) #blk=black image of size 500 * 500 pixels
    s1='Traffic area = '+ str(ta)
    s2='Traffic increased = '+ str(ti)
    s3='Traffic percentage = '+ str(tp)+'%'
    cv2.putText(blk, s1, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(blk, s2, (0,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(blk, s3, (0,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    wl=wi[:,2:]
    ws= np.ones((1000, 2, 3), dtype=np.uint8) * 255
    cv2.line(ws, (0, 20 * int(tp)), (0, 1000), (0, 0, 0), 3)
    di= np.hstack((wl, ws)) #di=data image
    wi=di
    di=cv2.flip(di,0)
    cv2.imshow('Original', f3)
    cv2.imshow('Graph',di)
    cv2.imshow('Traffic flow',blk)
    cv2.imshow('Traffic', binary_difference)
    cv2.imshow('Frame', frame)
    #cv2.imshow('Frame1', frame1)
    cv2.imshow('Road Monitored', wf)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.imshow('Original video',f3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
