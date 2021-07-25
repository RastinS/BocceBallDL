import numpy as np
import cv2
import yaml
import imutils



def calibration(nx, ny, path, numexamples):
    objpoints = []
    imgpoints = []
    objp = np.zeros((nx*ny,3), np.float32)
    for _ in range(1):
        for i in range(1,numexamples+1):
            print(i)
            if i <= 4 :
                fname = path + '%s.png'%i
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                fname = path + '%s.png'%i
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            objp[:,:2] = np.mgrid[0:27,0:9].T.reshape(-1,2)
            imgp = corners

            cv2.drawChessboardCorners(img, (7,6), imgp, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

            if ret == True:
                imgpoints.append(imgp)
                objpoints.append(objp)

    return objpoints,imgpoints


def get_parametrs(nx, ny, train_path, n):
    objpoints,imgpoints = calibration(nx,ny,train_path,n)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,(1080, 3), None, None)
    save_parametes(ret, mtx, dist, rvecs, tvecs)
    return ret, mtx, dist, rvecs, tvecs


def save_parametes(ret, mtx, dist, rvecs, tvecs):
    data = {'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist()}
    with open("calibration_matrix.yaml", "w") as f:
         yaml.dump(data, f)


def load_parametrs(path):
    if path == '':
        with open("calibration_matrix.yaml", "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return data
    else:
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return data


def undistorter(img, data):
    mtx = data['camera_matrix']
    dist = data['dist_coeff']
    h, w =  img.shape[:2]

    mtx = np.array(mtx)
    dist = np.array(dist)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]

    height = 1088
    width = int((height / dst.shape[0]) * dst.shape[1])

    xScale = width / dst.shape[1]
    yScale = height / dst.shape[0]

    xSkew = int(x * xScale)
    ySkew = int(y * yScale)

    dst = cv2.resize(dst, (width, height))
    return dst, (xSkew, ySkew), (xScale, yScale)


def getFrame(cap):
    _, frame = cap.read()

    if frame is None:
        print('frame fetch failure')
        return 0

    # frame = imutils.resize(frame, 1000)
    # frame = frame[:, 100:900]

    return frame


def cropFrameCenter1(frame, newWidth,x):

    return frame[:, int(frame.shape[1]/2-newWidth/2+x):int(frame.shape[1]/2+newWidth/2+x), :]


def cropFrameCenter2(frame, newWidth,x):

    return frame[:, int(frame.shape[1]/2-newWidth/2+x):int(frame.shape[1]/2+newWidth/2+x), :]


def cropFrameCenter3(frame, newWidth,x):

    return frame[:, int(frame.shape[1]/2-newWidth/2+x):int(frame.shape[1]/2+newWidth/2+x), :]


def cropFrameCenter4(frame, newWidth,x):

    return frame[:, int(frame.shape[1]/2-newWidth/2+x):int(frame.shape[1]/2+newWidth/2+x), :]

def cropFrameWithPoints(frame, points):
    return frame[points[3]:points[1], points[2]:points[0], :]


# rtsp = "rtsp://admin:admin123456@192.168.5.190:554/main"
# cap = cv2.VideoCapture(rtsp)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# frame = getFrame(cap)
# frame = cropFrameCenter(frame,640)
# cv2.imwrite('ff.png',frame)

# path = "calibration_matrix.yaml"
# data = load_parametrs(path)
# frame = undistorter(frame,data)
# cv2.imshow('ff',frame)
# cv2.waitKey(0)

if __name__ == '__main__':
    nx = 40
    ny = 50
    objpoints = []
    imgpoints = []
    objp = np.zeros((nx*ny,3), np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img = cv2.imread('./chessBoard5.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('a', imutils.resize(gray, 700))
    cv2.waitKey(0)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        print(corners)
    else:
        print('No Corners')
        exit(0)

    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    imgp = corners
    corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)

    cv2.drawChessboardCorners(img, (nx,ny), corners2, ret)
    cv2.imshow('img', imutils.resize(img, 700))
    cv2.waitKey(0)
    