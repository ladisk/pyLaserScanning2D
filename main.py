import sys
import gui
import os
import DAQTask
import logging
import cv2
import pyuff
import frf_new as frf

import matplotlib.pyplot as plt
from PyQt4 import QtGui, QtCore
import numpy as np
from time import sleep
from camera import Camera
from mesh import Mesh
from scanningsystem import ScanningSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExampleApp(QtGui.QMainWindow, gui.Ui_MainWindow):
    def __init__(self, parent=None):
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        self.populate_comboboxes()
        self.populate_default_values()
        self.set_validator()

        self.number_of_calibration_images = 0

        self.button_folder_measurement.clicked.connect(self.browse_folder_measurement)
        self.button_folder_image.clicked.connect(self.browse_folder_image)
        self.button_start.clicked.connect(self.measure_multiple)
        self.button_calculate_camera_parameters.clicked.connect(self.calculate_camera_parameters)
        self.button_correct_image.clicked.connect(self.correct_image)

        # self.button_start_camera.clicked.connect(self.measure_all)
        self.button_camera_start.clicked.connect(self.camera_start)
        self.button_camera_stop.clicked.connect(self.camera_stop)
        self.button_camera_capture_image.clicked.connect(self.camera_capture)

        self.button_preview_mesh.clicked.connect(self.create_mesh)

        self.button_reset_mirror.clicked.connect(self.reset_mirror)

        self.button_test.clicked.connect(self.write_uff) #measure_multiple)

        self.button_set_to_one_point.clicked.connect(self.set_to_one_point)


######################### TEST FUNCTIONS #########################################
    def set_to_one_point(self):
        point = self.lineedit_one_point.text().strip()
        logger.info(point)
        point = point.split(",")
        point = list(float(x) for x in point)
        mirror = self.create_mirror()
        mirror.set_to_point(point)
        return True


    def test_image(self):
        """
        Test image capture
        :return:
        """
        # capture image corrected image
        image = self.camera.capture_corrected_image(crop=False)
        # find spot
        # return spots
        r = 500.0 / image.shape[1]
        dim = (500, int(image.shape[0] * r))
        # perform the actual resizing of the image and show it
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("resized", resized)
        #cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True

    def test_mirror(self):
        mirror = self.create_mirror()
        mesh = self.create_mesh()
        for point in mesh:
            mirror.set_to_point(point)
            sleep(1)
        return True

    def test_force(self):
        """
        Test force measurement
        :return:
        """
        # clear task
        try:
            task.clear_task()
        except:
            pass
        # create task
        task = DAQTask.DAQTask("velocity_force")
        # get data
        task.acquire()
        logger.info(task.data)
        plt.plot(task.data[1])
        plt.show()
        #open file
        # write data
        try:
            task.clear_task()
        except:
            pass

######################## CREATE OBJECTS #############################################

    def create_mirror(self):
        """
        Creates 2-way mirror from gui data
        :return:
        """
        mirror_task_name = (self.combobox_canal_horizontal.currentText(),
                             self.combobox_canal_vertical.currentText())
        distance_to_object = float(self.lineedit_distance_to_object.text())
        distance_between_mirrors = float(self.lineedit_distance_between_mirrors.text())
        mirror = ScanningSystem(mirror_task_name, distance_between_mirrors, distance_to_object)
        logger.info("mirror created")
        return mirror

    def reset_mirror(self):
        """
        Set mirror to (0,0)
        :return:
        """
        mirror = self.create_mirror()
        mirror.reset()
        return True

    def create_mesh(self):
        """
        Creates mesh
        :return: mesh_points
        """
        points = self.read_points()
        max_volume = float(self.lineedit_mesh_max_area.text())
        min_angle = float(self.lineedit_mesh_min_angle.text())
        if self.radiobutton_area_border.isChecked():
            from_ = "border"
        else:
            from_ = "points"
        mesh = Mesh(from_=from_, points=points, max_volume=max_volume, min_angle=min_angle, show=True)
        points = mesh.get_points()
        print(points)
        return points

    def create_sensor(self):
        """
        creates force and velocity senzor
        :return:
        """
        task = self.combobox_canal_measure.currentText()
        return task

###################### GUI FUNCTIONS ##########################################


    def populate_comboboxes(self):
        """ Fill comboboxes in gui with awailable data (task in DAQTask object)

        :return:
        """
        canals = [task.decode() for task in DAQTask.get_daq_tasks()]
        self.statusbar.showMessage(str(canals))
        self.combobox_canal_horizontal.addItems(canals)
        self.combobox_canal_vertical.addItems(canals)
        self.combobox_canal_measure.addItems(canals)

    def set_validator(self):
        """ Validate input fields in gui
        :return:
        """
        regex = QtCore.QRegExp("[0-9 ]+")
        number_validator = QtGui.QRegExpValidator(regex)
        # self.lineedit_origin_x.setValidator(number_validator)
        # self.lineedit_origin_y.setValidator(number_validator)

    def populate_default_values(self):
        """ Fill some default values in gui

        :return:
        """
        self.lineedit_distance_to_object.insert("0.864")
        self.lineedit_distance_between_mirrors.insert("0.014")

        self.combobox_canal_horizontal.setCurrentIndex(0)
        self.combobox_canal_vertical.setCurrentIndex(1)
        self.combobox_canal_measure.setCurrentIndex(2)

        self.lineedit_folder_measurement.insert(os.path.dirname(os.path.realpath(__file__))+
                                                "\\"+"data")
        self.lineedit_folder_image.insert(os.path.dirname(os.path.realpath(__file__))+
                                          "\\"+"images")

        default_points = "0, 0\n0.02, 0.02\n-0.01, 0.04\n-0.05, -0.05"
        self.textedit_define_points.setText(default_points)

        self.lineedit_mesh_max_area.insert("0.05")
        self.lineedit_mesh_min_angle.insert("30")

    def browse_folder_measurement(self):
        """ Define/pick directory for writing data

        :return:
        """
        self.lineedit_folder_measurement.clear()
        directory = QtGui.QFileDialog.getExistingDirectory(None, "Pick a folder")
        #directory = QtGui.QFileDialog.getExistingDirectory(self, "Pick a folder")

        if directory:
            self.lineedit_folder_measurement.setText(directory)

    def browse_folder_image(self):
        """ Define/pick directory for writing data

        :return:
        """
        self.lineedit_folder_image.clear()
        directory = QtGui.QFileDialog.getExistingDirectory(self, "Pick a folder")

        if directory:
            self.lineedit_folder_image.setText(directory)

    def read_points(self):
        """ Read data from input text edit field and cleans them.
        :return: list of tuples [(x, y)]
        """
        points = self.textedit_define_points.toPlainText()
        points = points.split('\n')
        points = [point.strip().strip(',') for point in points]
        points = [point.strip().replace(' ', '').split(',') for point in points]
        points = [tuple([float(point[0]), float(point[1])]) for point in points]
        return points

############################# MEASUREMENT #################################

    def measure_one(self, plot = False, save = False):
        """ Measurement at one point
        :return: measured data
        """
        task = self.combobox_canal_measure.currentText()
        t = DAQTask.DAQTask(task)
        t.acquire()
        data = t.data
        logger.info(str(data))
        if plot:
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(data[0])
            ax2.plot(data[1])
            plt.show()
        if save:
            file_path = self.lineedit_folder_measurement.text()
            file_name = self.lineedit_file_name.text()
            file = file_path + "\\" + file_name
            if file_name == '' or os.path.isfile(file):
                file = QtGui.QFileDialog.getOpenFileName(self, "Save File", "",)
        logger.info("plot shown")
        return data

    def measure_multiple(self):
        """
        measure all points
        :return:
        """
        mirror = self.create_mirror()
        mesh = self.create_mesh()
        for point in mesh:
            mirror.set_to_point(point)
            data = self.measure_one()
            self.write_data_to_file(data, point)
            # self.calc_h(data, 4*51.2e3, show=True)
            sleep(0.1)
            logger.info(point)
        return True

    def calc_h(self, data, length, show=True):
        # build model
        model = frf.FRF(
            sampling_freq=51.2e3,
            exc=None,
            resp=None,
            exc_type='f',
            resp_type='v',
            exc_window='Hamming',
            resp_window='Hamming',
            resp_delay=0.0015,
            weighting='None',
            n_averages=1,
            fft_len=None,
            nperseg=length / 40,  # must be integer
            noverlap=None,
            archive_time_data=False,
            frf_type='H1')

        # read data
        logger.info(data)
        velocity, force = data[0], data[1]
        # add data to model
        model.add_data_for_overlapping(force, velocity)
        # plot force and velocity spectrum
        time_axis = np.arange(len(velocity)) / model.sampling_freq
        frequency_axis = model.get_f_axis()

        force_spectrum = model.get_exc_spectrum()
        acceleration_spectrum = model.get_resp_spectrum()
        H1 = model.get_H1()

        if show:
            """
            plt.figure(0, figsize=(5, 3))
            plt.plot(time_axis, velocity)
            plt.title("Hitrost")
            plt.ylabel("hitrost [mm/s]")
            plt.xlabel("čas [s]")

            plt.figure(1, figsize=(5, 3))
            plt.plot(time_axis, force)
            plt.title("Sila")
            plt.ylabel("sila [N]")
            plt.xlabel("čas [s]")

            plt.figure(2, figsize=(5, 3))
            plt.plot(frequency_axis, force_spectrum)
            plt.title("Spekter sile")
            plt.ylabel("[N^2/Hz]")
            plt.xlabel("frekvenca [Hz]")
            plt.xlim(0, 1000)

            plt.figure(3, figsize=(5, 3))
            plt.plot(frequency_axis, acceleration_spectrum)
            plt.title("Spekter pospeška")
            plt.ylabel("[mm^2/s^4/Hz]")
            plt.xlabel("frekvenca [Hz]")
            plt.xlim(0, 1000)
            """

            plt.figure(4)
            plt.semilogy(frequency_axis, np.absolute(H1))
            plt.title("H1 amplituda")
            plt.ylabel("H1 [mm/s^2/N]")
            plt.xlabel("frekvenca [Hz]")
            plt.xlim(0, 1000)
            plt.ylim(0, 1e6)
            plt.show()

            plt.figure(5)
            plt.plot(frequency_axis, np.angle(H1, deg=True))
            plt.title("H1 faza")
            plt.ylabel("H1 [°]")
            plt.xlabel("frekvenca [Hz]")
            plt.xlim(0, 1000)
            plt.show()

        return frequency_axis, H1

####################### CAMERA ###################################################

    def camera_start(self):
        self.camera = Camera(8000)
        self.camera.start()
        logger.info("camera started")

    def camera_capture(self):
        image = self.camera.capture_image(corrected=False, crop=False)
        image_path = self.lineedit_folder_image.text()
        image_name = "image_"+str(self.number_of_calibration_images) + ".jpg"
        cv2.imwrite(image_path+"\\"+image_name, image)
        self.number_of_calibration_images += 1

    def camera_capture_multiple(self):
        number_of_calibration_images = 13
        for i in range(number_of_calibration_images):
            logger.info("image capture loop started")
            self.camera_capture()
            logger.info("image captured")
            sleep(3)

    def camera_stop(self):
        self.camera.stop()

    def calculate_camera_parameters(self):
        try:
            self.camera = Camera(8000)
        except:
            pass
        folder = QtGui.QFileDialog.getExistingDirectory(None, 'Select a folder:', 'C:\\',
                                                            QtGui.QFileDialog.ShowDirsOnly)
        self.camera.calibrate_camera((7,9), 30, folder, show=True)

    def correct_image(self, image=None):
        self.camera = Camera(8000)
        if not image:
            image = QtGui.QFileDialog.getOpenFileName()
        corrected_image = self.camera.correct_image(image)
        cv2.imwrite('calibresult.jpg', corrected_image)

############################### FILE ###################################################
    def write_uff(self, path):
        # create file
        try:
            os.remove('test_nb.uff')
        except:
            pass
        uffwrite = pyuff.UFF(path + ".uff")
        number_of_points = len(os.listdir(path))

        data_151 = {'type': 151,
                    'model_name': 'model',
                    'description': 'vibration measurement',
                    'db_app': 'py2dlaserscanner',
                    'program': 'py2dlaserscanner'
                    }
        uffwrite._write_set(data_151, "overwrite")

        data_164 = {'type': 164,
                    'units_code': 9,
                    'length': 1,
                    'force': 1,
                    'temp': 1,
                    'temp_offset': 0
                    }
        uffwrite._write_set(data_164, "add")

        nodes = list(range(1, number_of_points + 1))
        cs = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 0]])
        for i in range(number_of_points):
            cs = np.append(cs, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]), axis=0)
        print(cs.shape)
        data_2420 = {'type': 2420,
                     'part_UID': 1,
                     'part_name': 'merjenec',
                     'cs_type': 0,
                     'cs_color': 8,
                     'nodes': nodes,
                     'local_cs': cs
                     }
        uffwrite._write_set(data_2420, "add")

        for i, filename in enumerate(os.listdir(path)):
            point = self.extract_point(filename)
            point_coordinates = np.array([[i + 1, point[0], point[1], 0]])
            if i == 0:
                grid_global = point_coordinates
            else:
                grid_global = np.append(grid_global, point_coordinates, axis=0)

        data_2411 = {'type': 2411,
                     'grid_global': grid_global
                     }
        uffwrite._write_set(data_2411, "add")

        # get all file names
        for i, filename in enumerate(os.listdir(path)):
            print(filename)
            # point = extract_point(path, filename)
            frequency_axis, H1 = self.calc_h(path + "\\" + filename, 4 * 51.2e3, show=True)
            H1 = H1.astype("complex64")
            # frequency_axis, H1 = frequency_axis[:10], H1[:10]
            data_58 = {'type': 58,
                       'func_type': 4,
                       'rsp_node': i + 1,
                       'rsp_dir': 3,
                       'ref_dir': 3,
                       'ref_node': 1,
                       'data': H1,
                       'x': frequency_axis,
                       'id1': 'id1',
                       'rsp_ent_name': 'ent_name'}
            uffwrite._write_set(data_58, "add")

    def extract_point(self, filename):
        file = filename.split("_")
        x = float(file[2])
        y = float(file[3][:-4])
        point = (x, y)
        return point

    def write_data_to_file(self, data, point=(0,0)):
        """
        Writes raw (numpy array) data to file
        :param data: numpy array to be written
        :param file_path: path and name of the file
        :param point: save point data in file name
        :return: True
        """
        #data = np.array(range(10))
        folder = self.lineedit_folder_measurement.text()
        filename_base = self.lineedit_file_name.text()
        if filename_base == '':
            filename_base = "test"
        # create_folder_if_not_exist
        if not os.path.exists(folder+"\\"+filename_base):
            os.makedirs(folder+"\\"+filename_base)
        file_name = filename_base+"_"+str(round(point[0],6))+\
                        "_"+str(round(point[1],6))+".txt"
        file_path = folder+"\\"+filename_base+"\\"+file_name
        logger.info(file_path)
        np.savetxt(file_path, data)
        return True

############################### MAIN ##################################################

def main():
    app = QtGui.QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    logger.info("Program started")
    app.exec_()


if __name__ == '__main__':
    main()

