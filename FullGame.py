from Config import readConfigFile
from CamUtils import *
import threading, time
from GameUtils import *
import imutils


def detectionThread(detectionInterval, calibrationMatrix, dataAccessLock, cropPointsConfig):
    global weightsPath, centers, lastProcessedImgs, cameraCaps, errorBalls, whitePresent, vLines, skew, scales
    model, imgsz, stride, device, half = loadModel(weightsPath, '', 1088)
    print(imgsz)

    while True:
        captures = []
        badImage = False
        # dataAccessLock.acquire()
        for i in range(len(cameraCaps)):
            success, img = cameraCaps[i].retrieve()
            if not success:
                print('bad frame')
                badImage = True
            # frame = cv2.imread('%d.jpg'%i)
            captures.append(img)
        # dataAccessLock.release()
        
        if badImage:
            continue

        images = []
        for frame in captures:
 
            frame, skew, scales = undistorter(frame, calibrationMatrix)
            # print(frame.shape)

            images.append(frame)
            
        for i in range(len(images)):
            frame = images[i]
            if i == 0:
                # frame = cropFrameCenter1(frame,800,17)
                # frame = cropFrameCenter3(frame,750,80)
                points = cropPointsConfig['cropPoints'][0]
                frame = cropFrameWithPoints(frame, points)
                pass
            elif i ==1:
                frame = cropFrameCenter2(frame,800,5)
                pass
            elif i ==2:
                frame = cropFrameCenter3(frame,800,10)
                pass
            elif i ==3:
                # frame = cropFrameCenter4(frame,350,50)
                frame = cropFrameCenter3(frame,750,80)
                pass

            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame = increase_brightness(frame, value=40)
            # print(frame.shape)
            # cv2.imwrite('test.png',frame)
            images[i] = frame
        lastProcessedImgs = images
        dataset = LoadImages(images, img_size=imgsz, stride=stride)

        res = getObjectCenters(model, dataset, imgsz, device, half, augment=False)

        res, vLines = extractVlines(res)

        whitePresent = False
        centers = []
        centerRadius = res

        for frameBall in res:
            frame = []
            for ball in frameBall:
                frame.append((ball[0], ball[1]))
                if ball[0] == 'white':
                    whitePresent = True
            centers.append(frame)


        # errorBalls = checkLineErrors(centerRadius, vLines)
        centers = (centers)


def frameFetchThread():
    global cameraCaps, dataAccessLock

    i = 1
    while True:
        # dataAccessLock.acquire()
        for i in range(len(cameraCaps)):
            cameraCaps[i].grab()
            # if i == 4:
            #     frame = cv2.imread('%d.jpg'%i)
            # else:
            #     frame = cv2.imread('%d.png'%i)
        # dataAccessLock.release()
        
 

if __name__ == "__main__":
    config = readConfigFile('config.json')
    cropPointsConfig = readConfigFile('cropConfigTest.json')

    images = [None] * config['numberOfCameras']
    centers = []
    lastProcessedImgs = [None] * config['numberOfCameras']
    detectionInterval = 0.2
    dataAccessLock = threading.Lock()
    cameraCaps = []
    frames = []
    for rtspAddress in config['cameraRTSPs']:
        cap = cv2.VideoCapture(rtspAddress)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cameraCaps.append(cap)

    calibrationMatrix = load_parametrs(config['calibrationMatrixAddress'])

    weightsPath = config['weightsAddress']

    skew = 0
    scales = 0
    fetchThread = threading.Thread(target=frameFetchThread)
    fetchThread.daemon = True
    fetchThread.start()

    detectThread = threading.Thread(target=detectionThread, args=[detectionInterval, calibrationMatrix, dataAccessLock, cropPointsConfig, ])
    detectThread.daemon = True
    detectThread.start()

    detectedBalls = []
    lastDetected = []
    lastBallColor = ''
    vLines = []
    winnerColor = ''
    errorBalls = []
    whitePresent = False
    time.sleep(10)
    while True:
        if len(lastProcessedImgs) > 0 and lastProcessedImgs[0].any() != None:
            im0 = lastProcessedImgs[0].copy()
            # im1 = lastProcessedImgs[1].copy()
            # im2 = lastProcessedImgs[2].copy()
            # im3 = lastProcessedImgs[3].copy()
            if len(centers) > 0:
                for className, coord in centers[0]:
                    if(className != 'vline'):
                        # im0 = cv2.circle(im0, (int(coord[0]), int(coord[1])), radius=1, color=(0, 0, 255), thickness=-1)
                        c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[0]+5), int(coord[1]+5))
                        tl = 1 or round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1
                        tf = 1  # font thickness
                        t_size = cv2.getTextSize(className, 0, fontScale=tl / 3, thickness=tf)[0]
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                        cv2.rectangle(im0, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
                        cv2.putText(im0, className, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                    else:
                        im0 = cv2.rectangle(im0, pt1=(int(coord[0]), int(coord[1])), pt2=(int(coord[2]), int(coord[3])), color=(0,0,255), thickness=1)
                    for coord in vLines[0]:
                        im0 = cv2.rectangle(im0, pt1=(int(coord[0]), int(coord[1])), pt2=(int(coord[2]), int(coord[3])), color=(0,0,255), thickness=1)

                # for className, coord in centers[1]:
                #     if(className != 'vline'):
                #         # im1 = cv2.circle(im1, (int(coord[0]), int(coord[1])), radius=1, color=(0, 0, 255), thickness=-1)
                #         c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[0]+5), int(coord[1]+5))
                #         tl = 1 or round(0.002 * (im1.shape[0] + im1.shape[1]) / 2) + 1
                #         tf = 1  # font thickness
                #         t_size = cv2.getTextSize(className, 0, fontScale=tl / 3, thickness=tf)[0]
                #         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                #         cv2.rectangle(im1, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
                #         cv2.putText(im1, className, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                #     # else:
                #     #     im1 = cv2.rectangle(im1, pt1=(int(coord[0]), int(coord[1])), pt2=(int(coord[2]), int(coord[3])), color=(0,0,255), thickness=1)
                #     # for coord in vLines[1]:
                #     #     im1 = cv2.rectangle(im1, pt1=(int(coord[0]), int(coord[1])), pt2=(int(coord[2]), int(coord[3])), color=(0,0,255), thickness=1)

                # for className, coord in centers[2]:
                #     if(className != 'vline'):
                #         # im1 = cv2.circle(im1, (int(coord[0]), int(coord[1])), radius=1, color=(0, 0, 255), thickness=-1)
                #         c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[0]+5), int(coord[1]+5))
                #         tl = 1 or round(0.002 * (im2.shape[0] + im2.shape[1]) / 2) + 1
                #         tf = 1  # font thickness
                #         t_size = cv2.getTextSize(className, 0, fontScale=tl / 3, thickness=tf)[0]
                #         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                #         cv2.rectangle(im2, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
                #         cv2.putText(im2, className, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                #     # else:
                #     #     im2 = cv2.rectangle(im2, pt1=(int(coord[0]), int(coord[1])), pt2=(int(coord[2]), int(coord[3])), color=(0,0,255), thickness=1)
                #     for coord in vLines[2]:
                #         im2 = cv2.rectangle(im2, pt1=(int(coord[0]), int(coord[1])), pt2=(int(coord[2]), int(coord[3])), color=(0,0,255), thickness=1)

                # for className, coord in centers[3]:
                #     if(className != 'vline'):
                #         # im1 = cv2.circle(im1, (int(coord[0]), int(coord[1])), radius=1, color=(0, 0, 255), thickness=-1)
                #         c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[0]+5), int(coord[1]+5))
                #         tl = 1 or round(0.002 * (im3.shape[0] + im3.shape[1]) / 2) + 1
                #         tf = 1  # font thickness
                #         t_size = cv2.getTextSize(className, 0, fontScale=tl / 3, thickness=tf)[0]
                #         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                #         cv2.rectangle(im3, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
                #         cv2.putText(im3, className, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                #     # else:
                #     #     im3 = cv2.rectangle(im3, pt1=(int(coord[0]), int(coord[1])), pt2=(int(coord[2]), int(coord[3])), color=(0,0,255), thickness=1)
                #     for coord in vLines[3]:
                #         im3 = cv2.rectangle(im3, pt1=(int(coord[0]), int(coord[1])), pt2=(int(coord[2]), int(coord[3])), color=(0,0,255), thickness=1)
                        
            # if errorBalls != None:
            #     for obj in errorBalls:
            #         print('Line error ball color : ', obj[0], ' in frame ', 'left' if obj[2] == 0 else 'right')
            #         if obj[2] == 0:
            #             coord = obj[1]
            #             im0 = cv2.circle(im0, (int(coord[0]), int(coord[1])), radius=5, color=(0, 0, 0), thickness=-1)
            #         elif obj[2] == -1:
            #             coord = obj[1]
            #             im3 = cv2.circle(im3, (int(coord[0]), int(coord[1])), radius=5, color=(0, 0, 0), thickness=-1)
            #     print('\n')
            # cv2.imshow('a', imutils.resize(cv2.hconcat([im0]), width=700))
            # im0 = imutils.resize(im0, width=700)
            x_center = int(im0.shape[1]/2 + 42)
            y_center = int(im0.shape[0]/2 + 3)
            im0 = cv2.circle(im0, (x_center, y_center), radius=5, color=(255,255,255), thickness=-1)
            cv2.imshow('a', imutils.resize(im0, height=700))
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                # for cap in cameraCaps:
                #     print('here')
                #     cap.release()
                realPos = computeAbsPositions(centers, im0.shape, skew, scales)
                if len(realPos) > 0:
                    dist = sortByDistance(realPos)

                    print('\n')
                    i = 1

                    for obj in dist:
                        print(i,obj[0])
                        i += 1
                    
                    print('winner: ', dist[0][0])
                break

        else:
            print('no frames processed')
        
        
        # cv2.waitKey(1)
        # if cv2.waitKey(1) & 0xFF == ord('s'):
            # break

    