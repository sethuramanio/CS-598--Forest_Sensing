# file of functions that can fill the isValid argument. 
# All functions require inputs:
# Sample s to test
# samples past samples
import numpy as np
import cv2


class Valids:
    def __init__(self, img, visibilityWindow):
        self.img = img
        # Calculate the water obstacles
        self.water = self.getWater()
        self.visibilityWindow = visibilityWindow

    def alwaysTrue(self, s, samples):
        return True
    
    def getWater(self):
        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # Identify blue pixels. 
        blue_low = np.array([90, 50, 70])
        blue_high = np.array([128, 255, 255])
        blue_pixels = cv2.inRange(img_hsv, blue_low, blue_high)

        return blue_pixels

    # Returns true if s is not on water, false otherwise. 
    def notOnWater(self, s, samples):
        x, y = s
        return self.water[x, y] == 0
    
    def notOnWater_distanced(self, s, samples):
        x, y = s
        if not self.notOnWater(s, samples):
            return False

        if len(samples) == 0:
            return True

        # check our nearest sample. Is it too close?
        nearest, dist = self.kNearestNeighbors(samples, s, k=1)
        if dist[0] < self.visibilityWindow/2:
            return False
        return True
    
    # I am too lazy to figure out an intelligent way around a circular import. 
    # Hence, copy/paste. 
    def kNearestNeighbors(self, vertices, v, k=5):
        if len(vertices) == 0:
            return [k], [0]
        if len(vertices) < k:
            k = len(vertices)-1

        samples_x_coords = np.array(vertices)[:,0]
        samples_y_coords = np.array(vertices)[:,1]
        s_x_coords = np.ones(len(vertices)) * v[0]
        s_y_coords = np.ones(len(vertices)) * v[1]
        euclideans = np.sqrt(np.square(samples_x_coords-s_x_coords)+np.square(samples_y_coords-s_y_coords))
        
        euclideans[np.where(euclideans == 0)] = np.inf # don't pick current element as its nearest neighbor

        min_indices = euclideans.argsort()[:k]

        verts = np.empty(len(vertices), dtype=object)
        verts[:] = vertices
        return verts[min_indices], euclideans[min_indices]
