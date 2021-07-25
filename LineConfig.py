import cv2
import argparse
from operator import xor


def callback(value):
    pass


def setup_trackbars(height=480, width=640):
    cv2.namedWindow("Trackbars", 0)

    cv2.createTrackbar('x1','Trackbars',0,width,callback)
    cv2.createTrackbar('y1','Trackbars',0,height,callback)
    cv2.createTrackbar('x2','Trackbars',0,width,callback)
    cv2.createTrackbar('y2','Trackbars',0,height,callback)


def get_trackbar_values():
    values = []

    x1 = cv2.getTrackbarPos('x1','Trackbars')
    y1 = cv2.getTrackbarPos('y1','Trackbars')
    x2 = cv2.getTrackbarPos('x2','Trackbars')
    y2 = cv2.getTrackbarPos('y2','Trackbars')

    return (x1, y1, x2, y2)


def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

def rescale_frame(frame, percent=400):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


def main():
    camera = cv2.VideoCapture(2)
    change_res(camera, 300, 200)
    camera.set(cv2.CAP_PROP_FPS, 120)

    setup_trackbars()

    while True:
        ret, image = camera.read()
        image = cv2.resize(image, (640, 480))
        print(image.shape)
        if not ret:
            break

        x1, y1, x2, y2 = get_trackbar_values()

        print(x1, x2, y1, y2)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1) 
        # image = rescale_frame(image)
  
        cv2.imshow("1", rescale_frame(image))

        if cv2.waitKey(1) & 0xFF is ord('q'):
            break


if __name__ == '__main__':
    main()