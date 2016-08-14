import numpy as np
import random

def random_weight():
    return random.uniform(-1, 1)

class Features(object):

    def __init__(self, fname=None):
        self.i = 1
        self.fdict = dict()
        self.fdict["empty_feature"] = 0
        self.add_feature = self.__build_features
        if fname:
            self.fdict = dict()
            with open(fname, "r") as infile:
                for line in infile:
                    fid, f = line.split()
                    self.fdict[f] = int(fid)
            self.add_feature = self.__return_only

    def __return_only(self, feature):
        return self.fdict.get(feature, None)

    def __build_features(self, feature):
        if feature in self.fdict:
            f_index = self.fdict[feature]
        else:
            self.fdict[feature] = self.i
            f_index = self.i
            self.i += 1
        return f_index

    def get_voc(self):
        for k in sorted(self.fdict, key=self.fdict.get):
            yield str(self.fdict[k]) + " " +  k + "\n"

    def generate_weights(self):
        for _ in self.fdict:
            yield random_weight()

class Vectors(object):

    def __init__(self):
        self.i = 0
        self.vecdict = dict()

    def add_vector(self, id_set):
        if id_set in self.vecdict:
            vec_index = self.vecdict[id_set]
        else:
            self.vecdict[id_set] = self.i
            vec_index = self.i
            self.i += 1
        return vec_index

    def get_voc(self):
        for k in sorted(self.vecdict, key=self.vecdict.get):
            yield str(self.vecdict[k]) + " " + " ".join(map(str, k)) + "\n"


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

