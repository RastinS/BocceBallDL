import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import math

from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized


from CamUtils import *
from GameUtils import *
from Config import *

from numpy.linalg import inv
from pylab import array, plot, show, axis, arange, figure, uint8 

class LoadImages:
    def __init__(self, images, img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.images = images
        self.nf = len(images)
        self.mode = 'image'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        img0 = self.images[self.count]

        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        self.count += 1
        return img, img0

    def __len__(self):
        return self.nf


def loadModel(weights, device, imgsz):
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)
    if half:
        model.half()
    
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    return model, imgsz, stride, device, half


def getObjectCenters(model, dataset, imgsz, device, half, conf_thres=0.4, iou_thres=0.45, augment=False, agnostic_nms=False, classes=None):
    names = model.module.names if hasattr(model, 'module') else model.names

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    results = []
    for img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=augment)[0]

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        imRes = []
        for i, det in enumerate(pred):
            s, im0, frame = '', im0s, getattr(dataset, 'frame', 0)

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape)

                for *xyxy, conf, cls in reversed(det):
                    if names[int(cls)] != 'vline': 
                        xRadius = abs((int(xyxy[2]) - int(xyxy[0]))/2)
                        yRadius = abs((float(xyxy[3]) - float(xyxy[1]))/2)
                        center = ((float(xyxy[0]) + float(xyxy[2]))/2, (float(xyxy[1]) + float(xyxy[3]))/2)
                        imRes.append((names[int(cls)], center, yRadius))
                    else:
                        imRes.append(('vline', (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])), 0))

        results.append(imRes)
    
    return results


def computeAbsPositions(objects, frameShape, skew, scales):
    data = load_parametrs('')
    chessBoardSize = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if chessBoardSize == 10:
        nx = 20
        ny = 24
        chessBoard = cv2.imread('chessBoard10.jpg')
        chessBoard = cv2.cvtColor(chessBoard, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(chessBoard, (nx, ny), None)
        corners2 = cv2.cornerSubPix(chessBoard,corners,(11,11),(-1,-1),criteria)
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    else:
        nx = 40
        ny = 50
        chessBoard = cv2.imread('chessBoard5.jpg')
        chessBoard = cv2.cvtColor(chessBoard, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(chessBoard, (nx, ny), None)
        corners2 = cv2.cornerSubPix(chessBoard,corners,(5,5),(-1,-1),criteria)
        corners2 = np.asarray(corners2)
        newarr = np.squeeze(corners2)
        newarr = np.concatenate((newarr, np.zeros((nx*ny, 1))), axis=1)
        objp = np.zeros((ny*nx,2), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    mtx = np.array(data['camera_matrix'])
    dist = np.array(data['dist_coeff'])

    objectspoints = []
    objectstype = []

    # for i in range(len(objects)):
    for i in range(1):
        for obj in objects[i]: 
            if obj[0] == 'vline':
                continue
            objectstype.append(obj[0])
            objectspoints.append(obj[1])

    objectspoints = np.array(objectspoints).astype(np.float64)
    objectspointsnew = np.zeros((objectspoints.shape[0],3))
    for i in range(objectspoints.shape[0]):
        objectspointsnew[i] = np.append(objectspoints[i],0)

    objectspoints = objectspointsnew.reshape(-1,3)

    (_, rotation_vector, translation_vector) = cv2.solvePnP(newarr, objp, mtx, np.array([]))


    x_center = frameShape[1]/2 + 40
    y_center = frameShape[0]/2 + 3
    center = [x_center, y_center, 0]

    for i in range(objectspoints.shape[0]):
        objectspoints[i] -= center
    (objectspoints, _) = cv2.projectPoints(objectspoints, rotation_vector, translation_vector, mtx, np.array([]))

    newobjects = []
    
    for i in range(objectspoints.shape[0]):
        newobjects.append([objectstype[i],objectspoints[i][0]])

    return newobjects


def sortByDistance(realPositions):
    whiteBallR = 2
    colorBallR = 4.5
    cameraHeight = 295

    coloredCoeff = (cameraHeight - colorBallR) / cameraHeight
    whiteCoeff = (cameraHeight - whiteBallR) / cameraHeight

    if len(realPositions) == 0:
        return
    whiteIdx = 0
    while(realPositions[whiteIdx][0] != 'white'):
        whiteIdx +=1
        if whiteIdx == len(realPositions):
            return

    whiteAlpha = math.atan2(abs(realPositions[whiteIdx][1][1]), abs(realPositions[whiteIdx][1][0]))
    whiteDistNorm = math.sqrt(math.pow(realPositions[whiteIdx][1][0], 2) + math.pow(realPositions[whiteIdx][1][1], 2)) * whiteCoeff

    whiteX = whiteDistNorm * math.cos(whiteAlpha) if realPositions[whiteIdx][1][0] > 0 else -1 * whiteDistNorm * math.cos(whiteAlpha)
    whiteY = whiteDistNorm * math.sin(whiteAlpha) if realPositions[whiteIdx][1][1] > 0 else -1 * whiteDistNorm * math.sin(whiteAlpha)

    distances = []
    for i in range(len(realPositions)):
        if i == whiteIdx:
            continue
        objType = realPositions[i][0]
        ballX = realPositions[i][1][0]
        ballY = realPositions[i][1][1]

        alpha = math.atan2(abs(ballY), abs(ballX))
        distNorm = math.sqrt(math.pow(ballX, 2) + math.pow(ballY, 2)) * coloredCoeff
        
        ballNewX = distNorm * math.cos(alpha) if ballX > 0 else -1 * distNorm * math.cos(alpha)
        ballNewY = distNorm * math.sin(alpha) if ballY > 0 else -1 * distNorm * math.sin(alpha)

        xDiff = ballNewX - whiteX
        yDiff = ballNewY - whiteY
        distance = np.sqrt(math.pow(xDiff, 2) + math.pow(yDiff, 2))

        distances.append((objType, distance))
    distances.sort(key=lambda x:x[1])

    return distances


def getWhiteBall(objects):
    for obj in objects:
        if obj[0] == 'white':
            return obj
    return None

def checkLineErrors(centerRadius, vLines):
    vLinesL = vLines[0]
    vLinesR = vLines[-1]

    errorBalls = []

    i = 0
    while (i + 1 < len(vLinesL)):
        if vLinesL[i + 1][2] - vLinesL[i][2] < 50:
            i += 1
        else:
            break
    if vLinesL != None:
        for obj in centerRadius[0]:
            if obj[0] == 'vline':
                continue
            if obj[1][0] - obj[2] < vLinesL[i][2]:
                errorBalls.append((obj[0], obj[1], 0))


    i = len(vLinesR) - 1
    while (i - 1 >= 0):
        if vLinesR[i][0] - vLinesR[i - 1][0] < 50:
            i -= 1
        else:
            break
    if vLinesR != None:
        for obj in centerRadius[-1]:
            if obj[0] == 'vline':
                continue
            if obj[1][0] + obj[2] > vLinesR[i][0]:
                errorBalls.append((obj[0], obj[1], -1))

    return errorBalls

def extractVlines(frameObjects):
    vLines = []
    for frame in frameObjects:
        frameVLines = []
        for obj in frame:
            if obj[0] == 'vline':
                frameVLines.append(obj[1])
                frame.remove(obj)
        frameVLines.sort(key=lambda x:x[0])
        vLines.append(frameVLines)
    return frameObjects, vLines


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    
    lim = 215
    value = 40
    s[s > lim] = 255
    s[s <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

if __name__ == '__main__':
    model, imgsz, stride, device, half = loadModel('weights/best.pt', '', 1088)
    config = readConfigFile('config.json')
    cropPointsConfig = readConfigFile('cropConfigTest.json')
    calibrationMatrix = load_parametrs(config['calibrationMatrixAddress'])

    im0 = cv2.imread('../30.jpg')
    im1 = cv2.imread('./0.jpg')

    captures = []
    images = []
    captures.append(im0)
    captures.append(im1)

    for frame in captures:
        frame, _, _ = undistorter(frame, calibrationMatrix)
        images.append(frame)
        
    for i in range(len(images)):
        frame = images[i]
        # if i == 0:
        #     frame = cropFrameCenter1(frame,350,17)
        #     pass
        # elif i ==1:
        #     frame = cropFrameCenter2(frame,350,5)
        #     pass
        # elif i ==2:
        #     frame = cropFrameCenter3(frame,350,5)
        #     pass
        if i == 0:
            # frame = cropFrameCenter4(frame,750,110)
            points = cropPointsConfig['cropPoints'][0].copy()
            # points[1] += 160
            # points[3] -= 0
            # points[2] += 50
            # points[0] += 60
            # print(points)
            frame = cropFrameWithPoints(frame, points)
            pass
        elif i == 1:
            # frame = cropFrameCenter4(frame,750,110)
            # points = cropPointsConfig['cropPoints'][0]
            # points[1] += 160
            # points[3] -= 40
            # points[2] += 50
            # points[0] += 60
            frame = cropFrameWithPoints(frame, cropPointsConfig['cropPoints'][0])
            pass

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame1 = frame.copy()
        frame = increase_brightness(frame, value=60)

        # cv2.imshow('a', imutils.resize(frame, 600))
        cv2.imshow('b', imutils.resize(frame1, 600))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # if i == 0:
        #     cv2.imwrite('chessBoard5.jpg', frame1)
        #     exit(0)
        # else:
        #     cv2.imwrite('chessBoard10.jpg', frame1)
        
        # print(frame.shape)
        # cv2.imwrite('test.png',frame)

        images[i] = frame

    dataset = LoadImages(images, img_size=imgsz, stride=stride)
    t0 = time.time()
    res = getObjectCenters(model, dataset, imgsz, device, half, augment=False)
    t1 = time.time() - t0

    
    whitePresent = False

    res, vLines = extractVlines(res)
    centers = []
    centerRadius = res
    for frameBall in res:
        frame = []
        for ball in frameBall:
            frame.append((ball[0], ball[1]))
            if ball[0] == 'white':
                whitePresent = True
        centers.append(frame)
    
    im0 = images[0]
    im1 = images[1]

    # print(res)
    # exit(0)
    # errorBalls = checkLineErrors(centerRadius, vLines)
    # if errorBalls != None:
    #     for obj in errorBalls:
    #         print('Line error ball color : ', obj[0], ' in frame ', 'left' if obj[2] == 0 else 'right')
    #         if obj[2] == 0:
    #             coord = obj[1]
    #             im0 = cv2.circle(im0, (int(coord[0]), int(coord[1])), radius=5, color=(0, 0, 0), thickness=-1)
    #         elif obj[2] == -1:
    #             coord = obj[1]
    #             im1 = cv2.circle(im1, (int(coord[0]), int(coord[1])), radius=5, color=(0, 0, 0), thickness=-1)
    # while(res[1][i][0] != 'white'):
    #     i += 1
    # del res[1][i]

    
    realPositions = computeAbsPositions(centers, im0.shape, 0, 0)
    # print(realPositions)
    distances = sortByDistance(realPositions)
    print('Distances calculated in %f seconds' % t1)
    for className, coord in centers[0]:
        if(className != 'vline'):
            # im0 = cv2.circle(im0, (int(coord[0]), int(coord[1])), radius=3, color=(0, 0, 255), thickness=-1)
            c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[0]+5), int(coord[1]+5))
            tl = 1 or round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1
            tf = 1  # font thickness
            t_size = cv2.getTextSize(className, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im0, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
            cv2.putText(im0, className, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        # else:
        #     im0 = cv2.rectangle(im0, pt1=(int(coord[0]), int(coord[1])), pt2=(int(coord[2]), int(coord[3])), color=(0,0,255), thickness=1)
        # for coord in vLines[0]:
        #     im0 = cv2.rectangle(im0, pt1=(int(coord[0]), int(coord[1])), pt2=(int(coord[2]), int(coord[3])), color=(0,0,255), thickness=1)

    for className, coord in centers[1]:
        if(className != 'vline'):
            # im1 = cv2.circle(im1, (int(coord[0]), int(coord[1])), radius=1, color=(0, 0, 255), thickness=-1)
            c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[0]+5), int(coord[1]+5))
            tl = 1 or round(0.002 * (im1.shape[0] + im1.shape[1]) / 2) + 1
            tf = 1  # font thickness
            t_size = cv2.getTextSize(className, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im1, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
            cv2.putText(im1, className, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        # else:
        #     im1 = cv2.rectangle(im1, pt1=(int(coord[0]), int(coord[1])), pt2=(int(coord[2]), int(coord[3])), color=(0,0,255), thickness=1)
        # for coord in vLines[1]:
        #     im1 = cv2.rectangle(im1, pt1=(int(coord[0]), int(coord[1])), pt2=(int(coord[2]), int(coord[3])), color=(0,0,255), thickness=1)

    print(distances)
    x_center = int(im0.shape[1]/2 + 40)
    y_center = int(im0.shape[0]/2 + 3)
    im0 = cv2.circle(im0, (x_center, y_center), radius=3, color=(255,255,255), thickness=-1)
    cv2.imshow('a', imutils.resize(im0, 700))
    # cv2.imshow('b', imutils.resize(im1, 700))
    # print(im0.shape)
    cv2.waitKey(0)    
