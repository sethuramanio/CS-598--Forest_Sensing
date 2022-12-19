import numpy as np
import cv2
from graph import Graph
from isvalids import Valids
import utils

RED_HSV = (0, 100, 100)
RED_RGB = (255, 0, 0)
RED_BGR = (0, 0, 255)

class Metrics:
    def __init__(self, img, visibilityWindow):
        self.img = img
        self.imglen = img.shape[0]
        self.imght = img.shape[1]
        # collect the water
        v = Valids(self.img, visibilityWindow)
        self.water = v.getWater()
        self.visibilityWindow = visibilityWindow
        self.traveling_coverage = int(visibilityWindow/2)

    def evaluate(self, g, path):
        assert(len(path) > 0)
        nsamples = g.nVertices()

        # length (proportional to fuel)
        length = self.pathLength(path)

        # % covered
        percentCovered = self.percentCoveredComplete(path)

        # land covered
        percentLandCovered = self.percentLandCoveredComplete(path)
        
        # % path land
        pathLandPercent = self.pathLandPercent(path)

        return nsamples, length, percentCovered, percentLandCovered, pathLandPercent

    def pathLength(self, path):

        length = 0
        for i in range(len(path)-1):
            length += utils.euclidean(path[i], path[i+1])
        
        return length

    def _colorPath(self, path):
        recolored_img = self.img.copy()
        
        # At every sample, color red. 
        for vertex in path:
            recolored_img = self.drawWindow(recolored_img, vertex)
        
        cv2.imwrite("window.png", recolored_img)

        return recolored_img

    def pathLandPercent(self, path):
        # Percent of the path that is land. 
        # Number of pixels on land / number of pixels total
        split_path = self._splitPath(path)
        recolored_img = self._colorPath(path)

        red_pixels_mask = cv2.inRange(recolored_img, RED_BGR, RED_BGR)
        
        land_space = np.bitwise_not(self.water)
        total_land = np.bitwise_and(red_pixels_mask, land_space)

        land_pixels = np.reshape(total_land, (self.imglen*self.imght))
        land_u, land_counts = np.unique(land_pixels, return_counts=True)
        land_dict = dict(zip(land_u, land_counts))

        red_u, red_counts = np.unique(red_pixels_mask, return_counts=True)
        red_dict = (dict(zip(red_u, red_counts)))

        return land_dict[255] / red_dict[255]

    # Percent of the whole land that is covered
    def percentCoveredComplete(self, path):
        # Idea: break the edges of the path into those with intermediate points
        # Then run percentCoveredAtVertices. 
        split_path = self._splitPath(path)
        return self.percentCoveredAtVertices(split_path)

    def percentLandCoveredComplete(self, path):
        split_path = self._splitPath(path)
        return self.percentLandCoveredAtVertices(split_path)

    def _splitPath(self, path):
        split_path = np.array([path[0]])
        for i in range(len(path)-1):
            p1x, p1y = path[i]
            p2x, p2y = path[i+1]
            if path[i] == path[i+1]:
                split_path = np.vstack((split_path, path[i]))
                continue

            rise = p2y-p1y
            run = p2x-p1x
            num_intervals = int(utils.euclidean(path[i], path[i+1])/self.traveling_coverage)
            if num_intervals == 0:
                split_path = np.vstack((split_path, path[i]))

            direction = np.array([run, rise])
            direction = direction / np.linalg.norm(direction)
            unit = (direction * self.traveling_coverage)#.astype(np.int)

            if unit[0] != 0:
                split_x = np.arange(p1x, p2x, unit[0])[:num_intervals].astype(np.int)
            else:
                split_x = p1x * np.ones(num_intervals)

            if unit[1] != 0:
                split_y = np.arange(p1y, p2y, unit[1])[:num_intervals].astype(np.int)
            else:
                split_y = p1y * np.ones(num_intervals)
            new_points = np.vstack((split_x, split_y)).T

            split_path = np.vstack((split_path, new_points))

        assert(len(split_path) >= len(path))
        return split_path.astype(np.int)

    def percentCoveredAtVertices(self, path):
        # recolor the image in red where we can see.
        # Then count pixels. 
        recolored_img = self._colorPath(path)
        nothing = cv2.bitwise_not(cv2.bitwise_or(self.water, cv2.bitwise_not(self.water)))
        return self._percentCovered(recolored_img, obstacles=nothing)

    def percentLandCoveredAtVertices(self, path):
        recolored_img = self._colorPath(path)
        colored_land_only = cv2.bitwise_and(recolored_img, recolored_img, mask=cv2.bitwise_not(self.water))
  
        return self._percentCovered(colored_land_only)

    def percentWaterCovered(self, path):
        recolored_img = self._colorPath(path)
        colored_water_only = cv2.bitwise_and(recolored_img, recolored_img, mask=self.water)
        
        return self._percentCovered(colored_water_only, obstacles=cv2.bitwise_not(self.water))

    def _percentCovered(self, img, obstacles=None):
        if obstacles is None:
            obstacles = self.water

        red_pixels_mask = cv2.inRange(img, RED_BGR, RED_BGR)
        free_space = np.bitwise_not(obstacles)
        total_seen = np.bitwise_and(red_pixels_mask, free_space)

        seen_pixels = np.reshape(total_seen, (self.imglen*self.imght))
        seen_u, seen_counts = np.unique(seen_pixels, return_counts=True)
        seen_dict = dict(zip(seen_u, seen_counts))

        free_pixels = np.reshape(free_space, (self.imglen*self.imght))
        free_u, free_counts = np.unique(free_pixels, return_counts=True)
        free_dict = dict(zip(free_u, free_counts))

        try:
            return seen_dict[255] / free_dict[255]
        except:
            return 0

    def getWindow(self, img, x, y):
        size_x = self.visibilityWindow
        size_y = self.visibilityWindow
    
        ptx, pty = x, y
        halfwidthx = int(size_x/2)
        halfwidthy = int(size_y/2)
        try:
            ex, why, _ = img.shape
        except:
            ex, why = img.shape
        ptx_min = min(max(0, ptx-halfwidthx), ex)
        pty_min = min(max(0, pty-halfwidthy), why)
        ptx_max = max(0, min(ex, ptx+halfwidthx))
        pty_max = max(0, min(why, pty+halfwidthy))

        return ptx_min, pty_min, ptx_max, pty_max

    def drawWindow(self, img, pt):
        ptx_min, pty_min, ptx_max, pty_max = self.getWindow(img, pt[0], pt[1])
        img = cv2.rectangle(img, (pty_min, ptx_min), (pty_max, ptx_max), RED_BGR, -1)
        return img
