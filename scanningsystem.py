import numpy as np
import DAQTask

class ScanningSystem():
    """ Scanning system (mirror) made from two mirrors: X and Y

    """
    def __init__(self, task_name, distance_between_mirrors=0.014,
                 distance_to_object=0.7, calibration_distance=0.868):
        """
        Arguments:
            task (list): list of tasks for setting angles
            distance_to_object (double): distance from second mirrro to the object
            distance_between_mirrors (double): distance between two mirrors (X and Y)

        """
        self.daq_task_x_name = task_name[0]
        self.daq_task_y_name = task_name[1]
        self.distance_to_object = distance_to_object
        self.distance_between_mirrors = distance_between_mirrors
        self.calibration_distance = calibration_distance
        # self.calibration_gap = 0.175
        # self.calibration_grid = np.array([])

    def set_to_angle(self, theta=(0,0)):
        """ Set mirrors to angle theta_x and theta_y
        :param theta: tuple (theta_x, theta_y)
        :param theta: tuple (theta_x, theta_y)
        :return: True if succesful
        """
        try:
            task_x.clear_task()
            task_y.clear_task()
        except:
            pass


        if abs(theta[0]) > 0.393 or abs(theta[1]) > 0.393:
            print("angle to big")
            return False

        theta_x = np.array([-theta[0], -theta[0]], dtype=np.float64)
        theta_y = np.array([theta[1], theta[1]], dtype=np.float64)

        task_x = DAQTask.DAQTask(self.daq_task_x_name)
        task_x.generate(theta_x,clear_task=True)

        task_y = DAQTask.DAQTask(self.daq_task_y_name)
        task_y.generate(theta_y, clear_task=True)
        return True

    def calculate_angle(self, point):
        """ return angle (theta_x, theta_y) for given point (x, y)
        :param point: tuple of (x, y)
        :return: tuple of (theta_x, theta_y) in rad

        """
        from math import atan, sqrt
        x = point[0]
        y = point[1]
        d = self.distance_to_object
        e = self.distance_between_mirrors

        theta_x = atan(x/(e + sqrt(d**2 + y**2)))
        theta_y = atan(y/d)

        return theta_x, theta_y

    def set_to_point(self, point):
        """ Sets mirror to desired point

        :param point: tuple (x, y) of point
        :return: True if angle was set, else False
        """
        theta_x, theta_y = self.calculate_angle(point)
        self.set_to_angle((theta_x, theta_y))
        return True

    def reset(self):
        """
        Set mirror to (0,0)
        :return:
        """
        self.set_to_angle()
        return True


if __name__ == "__main__":
    pass