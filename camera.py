import io
import socket
import struct
import numpy as np
import glob
import cv2
import pickle
from PIL import Image

class Camera():
    """ Camera module

    """
    def __init__(self, port, parameters = None):
        """
        Creates camera object, accepting incoming images on port
        :param port: port listening to incoming images
        :param parameters: camera calibration parameters
        """
        self.port = port
        try:
            self.parameters = pickle.load( open( "camera_parameters.p", "rb" ) )
        except:
            self.parameters = parameters

    def start(self):
        """
        # Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means all interfaces)
        :return: None
        """
        self.server_socket = socket.socket()
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(0)

        # Accept a single connection and make a file-like object out of it
        self.connection = self.server_socket.accept()[0].makefile('rb')
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop

    def stop(self):
        """
        Stops camera (listening on port)
        :return: None
        """
        self.connection.close()
        self.server_socket.close()

    def capture_image(self, corrected=False, crop=False):
        """
        Captures image from camera.
        :return: image
        """
        image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
        # Construct a stream to hold the image data and read the image
        # data from the connection
        self.image_stream = io.BytesIO()
        self.image_stream.write(self.connection.read(image_len))
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        self.image_stream.seek(0)
        image = Image.open(self.image_stream)
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        print("image returned", type(image), type(open_cv_image))
        if corrected:
            open_cv_image = self.correct_image(open_cv_image, crop=crop, image_type="opencv")
        return open_cv_image

    def calibrate_camera(self, shape, size, folder, show=False):
        """
        Calibrates camera
        :param shape: shape in the form of tuple (6,7) describing chessboard
        :param size: size of one rectangle in chessboard
        :param folder: folder where the images are located
        :param show: show images on the screen
        :return: tuple of calibration parameters
        """
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((shape[0] * shape[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:shape[1], 0:shape[0]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob(folder+"\\"+"*.jpg")

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (shape[1], shape[0]), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                if show:
                    img = cv2.drawChessboardCorners(img, (shape[1], shape[0]), corners2, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(500)

        cv2.destroyAllWindows()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        pickle.dump((ret, mtx, dist, rvecs, tvecs), open("camera_parameters.p", "wb"))
        self.parameters = ret, mtx, dist, rvecs, tvecs

    def correct_image(self, image, crop=False, image_type="file"):
        """
        Correct images with camera calibration data, optionally crop it.
        :param crop:
        :param image_type:
        :param image: image to correct
        :return: corrected image
        """
        if image_type == "file":
            img = cv2.imread(image)
        elif image_type == "opencv":
            img = image
        h, w = img.shape[:2]
        print(h, w)
        mtx = self.parameters[1]
        dist = self.parameters[2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        if crop:
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
        # cv2.imwrite('calibresult.png', dst)
        return dst


    def show_image(self, image, scale_factor):
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("window", image)
        cv2.waitKey()
        return True

    def laser_position(self, show = True):
        image = self.capture_image(corrected=True, crop=True)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        radius = 25
        grayg = cv2.GaussianBlur(gray, (radius, radius), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grayg)
        if show:
            new = image.copy()
            cv2.circle(new, maxLoc, radius, (255, 0, 0), 2)
            self.show_image(new, 0.4)
        return maxLoc
