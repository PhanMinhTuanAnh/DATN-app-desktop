from tkinter import *
from tkinter.ttk import Combobox
from tkinter import messagebox
import tkinter

import cv2
import PIL.Image, PIL.ImageTk # PIL.ImageTk: xử lý ảnh đưa vào tkinter

from time import sleep
from threading import Thread

from win32api import GetSystemMetrics

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

from utils import detector_utils as detector_utils
from utils import predictor_utils as predictor_utils

from utils.hiragana import hiragana
from utils.kanji import kanji
import cv2
import tensorflow as tf
import datetime
import argparse
import numpy as np
from scipy.signal import savgol_filter
from PIL import ImageFont, ImageDraw, Image
import numpy
import random


# Load Model ######################################################
detection_graph, sess = detector_utils.load_inference_graph()
# hiragana_model= predictor_utils.load_hiragana_model()
kanji_model= predictor_utils.load_kanji_model()
katakana_model=None
##############################################################

screenWidth = GetSystemMetrics(0)
screenHeight = GetSystemMetrics(1)
startWindowLocation = 0
window = Tk()
window.title("Tkinter OpenCV")
window.geometry(str(screenWidth) + "x" + str(screenHeight) + "+" + str(startWindowLocation) + "+" + str(startWindowLocation))

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, screenWidth)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, screenHeight)

# Thông số ########################################
start_time = datetime.datetime.now()
num_frames = 0
score_thresh = 0.8
im_width, im_height = (video.get(3), video.get(4))
num_hands_detect = 1
###################################################

# Variable ########################################
arrayDrawed = []

modifiedPoints = []
drawedPoints = []
isStart = False

# kalman ##########################################################
kalman = cv2.KalmanFilter(4, 2)
"""
    - dynamParams: This parameter states the dimensionality of the state
    - MeasureParams: This parameter states the dimensionality of the measurement
    - ControlParams: This parameter states the dimensionality of the control
    - vector.type: This parameter states the type of the created matrices that should be CV_32F or CV_64F
"""
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
# kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03 # lưu ý
# kalman.processNoiseCov = np.array([[3,0,0,0],[0,3,0,0],[0,0,3,0],[0,0,0,3]],np.float32) * 0.0003 # lưu ý
# kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 0.00003

check = False # to check hội tụ

isFirstPoint = False
isBacked = True
#################################################################
l = 2020 # left, right, top, bottom to crop image and predict
r = 0
t = 2020
b = 0
extractDataWeight = 0
#################################################################
pixelDraw = 8
pre_predict = -1

is_start_time_back = -1 # tính giờ cho back nếu lớn hơn 1,5s xóa hết
is_start_time_write = -1
###################################################

canvas_w = video.get(cv2.CAP_PROP_FRAME_WIDTH) // 4 * 3
canvas_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT) // 4 * 3

canvas = Canvas(window, width=canvas_w, height=canvas_h)
canvas.place(x=0, y=0)

selected_kanji_lbl_x = video.get(cv2.CAP_PROP_FRAME_WIDTH) // 4 * 3 + 20
selected_kanji_lbl_y = 20
selected_kanji_lbl = Label(window, text="", fg="red", font=("Arial",250)) # màu chữ
selected_kanji_lbl.place(x=selected_kanji_lbl_x, y=selected_kanji_lbl_y)

def predict_thread(img):
    global selected_kanji_lbl, kanji_model
    return selected_kanji_lbl.configure(text=kanji[predictor_utils.predict_all(img, kanji_model=kanji_model)])

def update_frame():
    global canvas, photo, bw, count, num_frames, elapsed_time, modifiedPoints, arrayDrawed, check, \
            is_start_time_back, is_start_time_write, pre_predict, pixelDraw, l, r, t, b, \
            isBacked, isFirstPoint
    ret, frame = video.read()
    frame = cv2.flip(frame, 1)

    # opencv BGR ảnh giao diện thì RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detection
    boxes, scores, classes = detector_utils.detect_objects(
            frame, detection_graph, sess)
    
    # DRAWWWW
    (left, right, top, bottom) = (boxes[0][1] * im_width, boxes[0][3] * im_width,
                                        boxes[0][0] * im_height, boxes[0][2] * im_height)
    
    if (classes[0] == 5 or classes[0] == 6 or classes[0] == 7) and scores[0] > score_thresh:
        is_start_time_back = -1
        ## kalman #####################################################
        p = np.array([np.float32(left+(right-left)/8), np.float32(top+(bottom-top)/8)])
        ptemp = np.array([np.float32(left+(right-left)/8), np.float32(top+(bottom-top)/8)]) # thằng này không đổi để chờ hội tụ
        coor = (int(left+(right-left)/8),int(top+(top-bottom)/8))
        isBacked = True
        kalman.correct(p)
        p = kalman.predict()
        # print(p)
        # if(classes[0] == 7 and (pre_predict == 5 or pre_predict == 6)): # kiểu thứ 3 đầu ngón qua bên phải
        #     p = np.array([np.float32(left+(right-left)/8*7), np.float32(top+(bottom-top)/8)])
        #     ptemp = np.array([np.float32(left+(right-left)/8*7), np.float32(top+(bottom-top)/8)]) # thằng này không đổi để chờ hội tụ
        #     coor = (int(left+(right-left)/8*7),int(top+(top-bottom)/8))
        #     while(abs(int(p[0])-coor[0]) > 0.1 and abs(int(p[1])-coor[1]) > 0.1):
        #         arrayDrawed.append((int(p[0]),int(p[1])))
        #         kalman.correct(ptemp)
        #         p = kalman.predict()
        while(abs(int(p[0])-coor[0]) > 0.1 and abs(int(p[1])-coor[1]) > 0.1 and check == False):
            kalman.correct(ptemp)
            p = kalman.predict()

        check = True
        cv2.line(frame, (int(p[0]),int(p[1])), (int(p[0]),int(p[1])), (255, 255, 0), 30)
        arrayDrawed.append((int(p[0]),int(p[1])))


    elif classes[0] == 4 and scores[0] > score_thresh:
        is_start_time_back = -1
        is_start_time_write = -1
        check = False
        isBacked = True
        isFirstPoint = False
        countPassedPoint = 0

        if(len(arrayDrawed) > 0): # có thì mới add được
            modifiedPoints.append(arrayDrawed)
        arrayDrawed = []

    elif ((classes[0] == 3 or classes[0] == 2)  and scores[0] > score_thresh): # check and stop
        
        is_start_time_write = -1
        is_start_time_back = -1
        check = False
        isBacked = True
        print(l,r,t,b)
        if(len(arrayDrawed) > 0): # có thì mới add được
            modifiedPoints.append(arrayDrawed)
        for modifiedPoint in modifiedPoints:
            for i in range(1,len(modifiedPoint)):
                if(l > modifiedPoint[i-1][0]):
                    l = modifiedPoint[i-1][0]
                if(r < modifiedPoint[i-1][0]):
                    r = modifiedPoint[i-1][0]
                if(t > modifiedPoint[i-1][1]):
                    t = modifiedPoint[i-1][1]
                if(b < modifiedPoint[i-1][1]):
                    b = modifiedPoint[i-1][1]
                # cv2.line(frame, modifiedPoint[i], modifiedPoint[i-1], (0, 255, 0), 15)
            if(l > modifiedPoint[len(modifiedPoint)-1][0]):
                l = modifiedPoint[len(modifiedPoint)-1][0]
            if(r < modifiedPoint[len(modifiedPoint)-1][0]):
                r = modifiedPoint[len(modifiedPoint)-1][0]
            if(t > modifiedPoint[len(modifiedPoint)-1][1]):
                t = modifiedPoint[len(modifiedPoint)-1][1]
            if(b < modifiedPoint[len(modifiedPoint)-1][1]):
                b = modifiedPoint[len(modifiedPoint)-1][1]
        if(r-l > 0 and b-t > 0):
            maxsize = r-l
            if r-l < b-t: 
                maxsize = b-t
            # elif r-l > b-t:
            #     maxsize = r-l
            
            extractDataWeight = maxsize // 15  #(11 hình sai 1)
            if extractDataWeight == 0:
                extractDataWeight = 1

            img = numpy.zeros([maxsize+30, maxsize+30, 3])
            
            for modifiedPoint in modifiedPoints:
                for i in range(1,len(modifiedPoint)):
                    if(maxsize == 0):
                        cv2.line(img, (modifiedPoint[i][0]-l+15,modifiedPoint[i][1]-t+15), (modifiedPoint[i-1][0]-l+15,modifiedPoint[i-1][1]-t+15), (255,255,255), extractDataWeight)
                    elif(maxsize == r-l):
                        addPad = ((r-l)-(b-t)) // 2
                        cv2.line(img, (modifiedPoint[i][0]-l+15,modifiedPoint[i][1]-t+15+addPad), (modifiedPoint[i-1][0]-l+15,modifiedPoint[i-1][1]-t+15+addPad), (255,255,255), extractDataWeight)
                    elif(maxsize == b-t):
                        addPad = ((b-t)-(r-l)) // 2
                        cv2.line(img, (modifiedPoint[i][0]-l+15+addPad,modifiedPoint[i][1]-t+15), (modifiedPoint[i-1][0]-l+15+addPad,modifiedPoint[i-1][1]-t+15), (255,255,255), extractDataWeight)
                    # print((modifiedPoint[i][0]-l,modifiedPoint[i][1]-t))
            cv2.imwrite('img.jpg', img)
            # print(hiragana[predictor_utils.predict_all(img, hiragana_model)])
            # print(kanji[predictor_utils.predict_all(img, kanji_model=kanji_model)])
            # selected_kanji_lbl.configure(text=kanji[predictor_utils.predict_all(img, kanji_model=kanji_model)])
            # đa luồng: 
            thread = Thread(target=predict_thread(img))
            thread.start()

        l = 2020 # left, right, top, bottom to crop image and predict
        r = 0
        t = 2020
        b = 0
        extractDataWeight = 0
        # cv2.imshow('st2',frame[t:b,l:r])
        # print(hiragana[predictor_utils.predict_all()])
        arrayDrawed = []
        modifiedPoints = []
        # isStart = False
        # check = False
    elif classes[0] == 1  and scores[0] > score_thresh:
        is_start_time_write = -1
        check = False
        if(is_start_time_back == -1): # nếu chưa back trước lần nào thì gán
            is_start_time_back = datetime.datetime.now()
            if(isBacked == True):
                if(len(arrayDrawed) > 0): # có thì mới add được # kiểm tra xem đã vẽ gì chưa để add vào trước khi xóa
                    modifiedPoints.append(arrayDrawed)
                    arrayDrawed = [] # pop rồi nhưng thằng này vẫn vẽ ??:D??
                if(len(modifiedPoints) > 0): # if empty không cần pop
                    modifiedPoints.pop(-1)
                isBacked = False
        else:
            if((datetime.datetime.now()-is_start_time_back).total_seconds()>1.5):
                arrayDrawed = []
                modifiedPoints = []

    pre_predict = classes[0]

    # print(modifiedPoints)
    
    for modifiedPoint in modifiedPoints:
        for i in range(1,len(modifiedPoint)):
            cv2.line(frame, modifiedPoint[i], modifiedPoint[i-1], (0, 255, 0), pixelDraw)

    for i in range(1,len(arrayDrawed)):
        cv2.line(frame, arrayDrawed[i], arrayDrawed[i-1], (0, 255, 0), pixelDraw) 
 
    
    detector_utils.draw_box_on_image(
            num_hands_detect, score_thresh, classes, scores, boxes, im_width, im_height, frame)
    num_frames += 1
    elapsed_time = (datetime.datetime.now() -
                        start_time).total_seconds()
    fps = num_frames / elapsed_time
    cv2.putText(frame, "FPS: "+ str(int(fps)), (20, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0,255,255))
    frame = cv2.resize(frame, dsize=None, fx=0.75, fy=0.75)
    
    # frame là mảng array numpy
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame)) # convert mảng thành image, xong từ image convert qua image tk
    # show
    canvas.create_image(0, 0, image=photo, anchor=tkinter.NW) # neo 
    
    window.after(1, update_frame)

update_frame()
window.mainloop()
