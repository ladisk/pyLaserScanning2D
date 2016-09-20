import numpy as np
import matplotlib.pyplot as plt

class Mesh():
    """
    Mesh modul
    """
    def __init__(self,from_,points, max_volume = 0.05, min_angle = 30, show=True):
        if from_ == "points":
            self.mesh_points = self.from_points(points)
        elif from_ =="border":
            self.mesh_points, self.mesh = self.from_border(points, max_volume, min_angle)
            if show:
                # Matplotlib's Triangulation module uses only linear elements, so use only first 3 columns when plotting
                pts, mesh = self.mesh_points, self.mesh
                elements = np.vstack(mesh.elements)  # (ntriangles, 6)-array specifying element connectivity
                plt.triplot(pts[:, 0], pts[:, 1], elements[:, :3])
                plt.plot(pts[:, 0], pts[:, 1],'ko')  # Manually plot all points including the ones at the midpoints of triangle faces
            plt.show()


    def from_points(self,points):
        return points

    def from_border(self,points, max_volume, min_angle):
        """ Creates mesh with from points (defined in gui)

        :return: Return 2d numpy array of mesh/points
        """

        import meshpy.triangle as triangle

        def round_trip_connect(start, end):
            result = []
            for i in range(start, end):
                result.append((i, i + 1))
            result.append((end, start))
            return result

        info = triangle.MeshInfo()
        info.set_points(points)
        info.set_facets(round_trip_connect(0, len(points) - 1))

        mesh = triangle.build(info, max_volume=max_volume, min_angle=min_angle)
        pts = np.vstack(mesh.points)  # (npoints, 2)-array of points
        return pts, mesh

    def get_points(self):
        """
        return points
        :return: points
        """
        return self.mesh_points

    def mesh_from_regular_grid(self):
        x = np.arange(-0.16, 0.17, 0.08)
        y = np.arange(-0.11, 0.12, 0.055)
        points = np.vstack(np.meshgrid(x, y)).reshape(2, -1).T
        return points
