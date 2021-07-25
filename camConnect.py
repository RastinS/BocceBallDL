import imutils
import time
import cv2
# from imagecorrection2 import load_parametrs
# from imagecorrection2 import undistorter
# from crop import *

# time.sleep(10)

# rtsp://192.168.1.188:554/ch01.264
cap1 = cv2.VideoCapture("rtsp://admin:admin123456@192.168.5.193:554/main")
cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# cap2 = cv2.VideoCapture('rtsp://192.168.5.191')
# cap3 = cv2.VideoCapture('rtsp://192.168.1.15:554/ch01.264')
# cap4 = cv2.VideoCapture('rtsp://192.168.1.243:554/ch01.264')

i = 100
# path = "calibration_matrix.yaml"
# data = load_parametrs(path)
while True:
    _, frame1 = cap1.read()
    # _, frame2 = cap2.read()
    # _, frame3 = cap3.read()
    # _, frame4 = 
    # cap4.read()

    if frame1 is None:
        print('frame fetch failure')
        continue



    # frame1 = imutils.resize(frame1, 1000)
    # frame1 = 
    # frame1[0:562, 0:980]
    # frame1 = cv2.flip(frame1,1)
    # frame1 = cv2.line(frame1, (int(frame1.shape[1]/2)-50, int(frame1.shape[0]/2)), (int(frame1.shape[1]/2)+50, int(frame1.shape[0]/2)), color=(255, 0, 0), thickness=3)
    # frame1 = cv2.line(frame1, (int(frame1.shape[1]/2), int(frame1.shape[0]/2)-50), (int(frame1.shape[1]/2), int(frame1.shape[0]/2)+50), color=(255, 0, 0), thickness=3)
    cv2.imshow('frame1', imutils.resize(frame1, width=1000))

    # frame2 = imutils.resize(frame2, 1000)
    # frame2 = frame2[0:562, 0:980]
    # frame2 = cv2.flip(frame2,1)
    # cv2.imshow('frame2', frame2)

    # frame3 = imutils.resize(frame3, 1000)
    # frame3 = frame3[0:562, 0:980]
    # frame3 = cv2.flip(frame3,1)
    # cv2.imshow('frame3', frame3)


    # frame4 = imutils.resize(frame4, 1000)
    # frame4 = frame4[0:562, 0:980]
    # frame4 = cv2.flip(frame4,1)
    # cv2.imshow('frame4', frame4)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('%d.jpg' % (i), frame1)
    # cv2.imwrite('%d.jpg' % (i), frame2)
    # cv2.imwrite('%d.jpg' % (3), frame3)
    # cv2.imwrite('%d.jpg' % (4), frame4)
        # i += 1
        break
    # cv2.imshow('frame1', frame1)
    # cv2.imshow('frame2', frame2)
    # cv2.imshow('frame3', frame3)
    # cv2.imshow('frame4', frame4)


    
    


    # frame1 = undistorter(frame1,data,1)
    # frame2 = undistorter(frame2,data,2)
    # frame3 = undistorter(frame3,data,3)
    # frame4 = undistorter(frame4,data,4)


    # cv2.imwrite('frame3.jpg', frame3)
    # cv2.imwrite('frame1.jpg', frame1)
    # cv2.imwrite('frame2.jpg', frame2)
    # cv2.imwrite('frame4.jpg', frame4)

    # frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
    # frame1 = cv2.rotate(frame1, cv2.ROTATE_180)

    # frame2 = cv2.rotate(frame2, cv2.ROTATE_90_CLOCKWISE)
    # frame2 = cv2.rotate(frame2, cv2.ROTATE_180)

    # frame3 = cv2.rotate(frame3, cv2.ROTATE_90_CLOCKWISE)
    # frame3 = cv2.rotate(frame3, cv2.ROTATE_180)

    # frame4 = cv2.rotate(frame4, cv2.ROTATE_90_CLOCKWISE)
    # frame4 = cv2.rotate(frame4, cv2.ROTATE_180)


    # cv2.imwrite('undist1.jpg', frame1)
    # cv2.imwrite('undist2.jpg', frame2)
    # # cv2.imwrite('frame3.jpg', frame3)
    # # cv2.imwrite('frame4.jpg', frame4)
    





    # frame1 = Cframe1(frame1)
    # frame2 = Cframe2(frame2)
    # frame3 = Cframe3(frame3)
    # frame4 = Cframe4(frame4)



    # final = np.concatenate((frame1,frame2),axis=1)
    # final = np.concatenate((final,frame3),axis=1)
    # final = np.concatenate((final,frame4),axis=1)



    # cv2.imwrite('pic/pic%s.jpg'%i,final)
    # i += 1
    # print(i)
    # break
    # time.sleep(5)


    
    


cap1.release()
# cap2.release()
# cap3.release()
# cap4.release()

# cv2.destroyAllWindows()
