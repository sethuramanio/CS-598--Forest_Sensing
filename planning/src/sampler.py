# returns one valid sample, uniform dist
import numpy as np

class Samplers:
    def __init__(self, img, visibilityWindow=100):
        self.img = img
        self.visibilityWindow = visibilityWindow

    def uniform(self, isValid, nsamples=1):
        samples = []
        nfailed = 0
        while len(samples) != nsamples and nfailed < nsamples:
            x = np.random.randint(0, high=self.img.shape[0])
            y = np.random.randint(0, high=self.img.shape[1])
            s = (x, y)
            if isValid(s, samples):
                samples.append(s)
            else:
                nfailed += 1
        return samples

    def grid(self, isValid, nsamples=1):
        spacing = nsamples
        size = self.visibilityWindow * spacing
        halfsize = int(size/2)
        x_coords = np.linspace(halfsize, self.img.shape[0]-halfsize, int(self.img.shape[0]/size),endpoint=True)
        
        np.concatenate((np.arange(halfsize, self.img.shape[0]-halfsize, size), [self.img.shape[0]-halfsize]))
        y_coords = np.linspace(halfsize, self.img.shape[1]-halfsize, int(self.img.shape[1]/size),endpoint=True)
        
        np.concatenate((np.arange(halfsize, self.img.shape[1]-halfsize, size), [self.img.shape[1]-halfsize]))

        all_x_coords, all_y_coords = np.meshgrid(x_coords, y_coords)
        all_x_coords = all_x_coords.flatten()
        all_y_coords = all_y_coords.flatten()

        samples = np.vstack((all_x_coords, all_y_coords)).astype(int).T
        samples = map(tuple, samples)
        valid_samples = []
        for sample in samples:
            if isValid(sample, samples):
                valid_samples.append(sample)

        return valid_samples

