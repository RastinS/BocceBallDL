import json
import time
import cv2
from CamUtils import *

def createConfigFile():
    config = {}
    config['numberOfCameras'] = int(input('Input number of cameras: '))

    print('Enter RTSP addresses of cameras in left-to-right order')
    config['cameraRTSPs'] = []
    for i in range(config['numberOfCameras']):
        rtsp = input('Enter camera number %d rtsp: ' % (i+1))
        config['cameraRTSPs'].append(rtsp)

    config['calibrationMatrixAddress'] = input('Enter calibration matrix file address: ')

    config['weightsAddress'] = input('Enter wights address: ')

    with open('config.json', 'w') as outfile:
        json.dump(config, outfile, indent=4)

    print('\nConfig file created!')


def callback(value):
    pass


def setup_trackbars(height=480, width=640):
    cv2.namedWindow("Trackbars", 0)

    cv2.resizeWindow('Trackbars', 500, 300)
    cv2.createTrackbar('x1', 'Trackbars', 0, width, callback)
    cv2.createTrackbar('y1', 'Trackbars', 0, height, callback)
    cv2.createTrackbar('x2', 'Trackbars', 0, width, callback)
    cv2.createTrackbar('y2', 'Trackbars', 0, height, callback)


def get_trackbar_values():
    x1 = cv2.getTrackbarPos('x1','Trackbars')
    y1 = cv2.getTrackbarPos('y1','Trackbars')
    x2 = cv2.getTrackbarPos('x2','Trackbars')
    y2 = cv2.getTrackbarPos('y2','Trackbars')

    return (x1, y1, x2, y2)


def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def cropMarksConfig():
    config = readConfigFile('config.json')
    calibrationMatrix = load_parametrs(config['calibrationMatrixAddress'])

    cropPointsConfig = {}
    cameraCaps = []
    cropPointsConfig['cropPoints'] = []
    # for rtspAddress in config['cameraRTSPs']:
    #     cap = cv2.VideoCapture(rtspAddress)
    #     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    #     cameraCaps.append(cap)
    
    cameraCaps.append(1)
    for cap in cameraCaps:
        # success = False
        # while success == False:
        #     success, frame = cap.read()

        frame = cv2.imread('./22.jpg')
        frame, _, _ = undistorter(frame, calibrationMatrix)

        setup_trackbars(frame.shape[0], frame.shape[1])

        x1, y1, x2, y2 = 0, 0, 0, 0
        while True:
            frame2 = frame.copy()
            x1, y1, x2, y2 = get_trackbar_values()

            print(x1, x2, y1, y2)
            frame2 = cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2) 
    
            cv2.imshow("1", rescale_frame(frame2))

            if cv2.waitKey(1) & 0xFF is ord('q'):
                break
        
        cropPointsConfig['cropPoints'].append((x1, y1, x2, y2))

    with open('cropConfigTest.json', 'w') as outfile:
        json.dump(cropPointsConfig, outfile, indent=4)



def readConfigFile(path):
    with open(path) as jsonFile:
        config = json.load(jsonFile)
    return config

if __name__ == '__main__':
    cropMarksConfig()