from collections import defaultdict
import numpy as np

class Features(object):
    # static and dynamic features
    # static ones are stored in subcorpus file so they don't need to be extracted again
    # dynamic ones need to be accounted for (reserve a spot in feature vector) and generate weights

    def __init__(self):
        self.feature_num = 0
        self.feature_dict = dict()

    def add(self, feat):
        if feat not in self.feature_dict:
            self.feature_dict[feat] = self.feature_num
            self.feature_num += 1
        return self.feature_dict[feat]

    def get_feat_id(self, feat):
        return self.feature_dict[feat]

    def get_voc(self):
        output = ""
        for k in sorted(self.feature_dict, key=self.feature_dict.get):
            output += str(self.feature_dict[k]) + "\t" + " ".join(map(str, k)) + "\n"
        return output

class MaxDict(object):

    def __init__(self):
        self.max = 0

    def add(self, v):
        if v > self.max:
            self.max = v
    def get(self):
        return self.max

def max_dict():
    return MaxDict()


def load_vecs(file_name):
    vec_ids = dict()
    infile = open(file_name, "r")
    for line in infile:
        els = line.split()
        vec_id = els[0]
        f_ids = map(int, els[1:])

        vec_ids[vec_id] = f_ids
    infile.close()
    return vec_ids


def load_weights(file_name):
    d_weights = []
    with open(file_name, "r") as infile:
        for line in infile:
            w_id, w = line.strip().split()
            d_weights.append(float(w))

    return np.array(d_weights)

def convoc_reader(fname):
    with open(fname, "r") as infile:
        for line in infile:
            t, con, max_I = line.split()
            max_I = int(max_I)
            yield t, con, max_I

