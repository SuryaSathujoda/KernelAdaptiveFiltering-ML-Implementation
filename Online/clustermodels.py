import models
import numpy as np
import copy

class NICE():
    def __init__(
        self,
        input,
        output,
        model,
        cluster_thresh,
        learning_step = 0.5,
        sigma = 1,
        quant_thresh = None
    ):
        self.learning_step = learning_step
        self.sigma = sigma
        self.cluster_thresh = cluster_thresh

        if(quant_thresh == None):
            self.clusters = [model(input, output, learning_step, sigma)]
            self.is_quant = 0
        else:
            self.clusters = [model(input, output, quant_thresh, learning_step, sigma)]
            self.is_quant = 1

        self.cluster_points = [input]
        self.cluster_centres = [input]
        self.cluster_size = [1]
        self.pred = [0]

    def calc_dist(self, new_input):
        diff = self.cluster_centres - new_input
        dist = np.einsum("ij,ij->i", diff, diff)
        min_pos = np.argmin(dist)
        return min_pos, dist[min_pos]

    def update_clusters(self, min_pos, new_input, create_new):
        if(create_new == 0):
            self.cluster_points[min_pos] = np.vstack((self.cluster_points[min_pos], new_input))
            cc_centre = self.cluster_centres[min_pos]
            cc_size = self.cluster_size[min_pos]
            self.cluster_centres[min_pos] = (cc_centre * cc_size + new_input) / (cc_size + 1)
            self.cluster_size[min_pos] += 1
        else:
            self.cluster_points.append(new_input)
            self.cluster_centres.append(new_input)
            self.cluster_size.append(1)

    def predict(self, min_pos, new_input, expected, create_new):
        if(create_new == 0):
            self.clusters[min_pos].update(new_input, expected)
            self.pred.append(self.clusters[min_pos].pred[-1])
        else:
            self.clusters.append(copy.deepcopy(self.clusters[min_pos]))
            self.clusters[-1].update(new_input, expected)
            self.pred.append(self.clusters[-1].pred[-1])

    def update(self, new_input, expected):
        min_pos, dist = self.calc_dist(new_input)
        if(dist < self.cluster_thresh):
            self.predict(min_pos, new_input, expected, 0)
            if(self.is_quant == 0 or self.clusters[min_pos].merge == 0):
                self.update_clusters(min_pos, new_input, 0)
        else:
            self.predict(min_pos, new_input, expected, 1)
            self.update_clusters(min_pos, new_input, 1)

    def name(self):
        return "NICE"
