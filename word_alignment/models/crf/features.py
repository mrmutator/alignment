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

class FeatureStore(object):

    def __init__(self, feature_file):
        self.index = 0
        self.lookup = dict()
        self.features = dict()
        self.read_feature_file(feature_file)


    def add_key(self, k):
        self.lookup[k] = defaultdict(set)
        self.lookup[k][None] = set(self.features.keys())


    def add_features(self, features):
        self.index += 1
        curr_id = self.index
        features = dict(features)
        for k in features:
            if k not in self.lookup:
                self.add_key(k)

        self.features[curr_id] = features
        for k in self.lookup:
            if k in features:
                self.lookup[k][features[k]].add(curr_id)
            else:
                self.lookup[k][None].add(curr_id)

    def read_feature_file(self, fn):
        collected = set()
        with open(fn, "r") as infile:
            for line in infile:
                if line.strip():
                    f_pairs = line.strip().split("\t")
                    f_pairs = [tuple(p.split(" ")) for p in f_pairs]
                    subfeatures = frozenset([(k, int(v)) for k, v in f_pairs])
                    if subfeatures not in collected:
                        self.add_features(subfeatures)
                        collected.add(subfeatures)

    def get_voc(self):
        voc = ""
        for fid in sorted(self.features):
            voc += "\t".join([str(fid)] + [k + " " + str(v) for k,v in self.features[fid].iteritems()]) + "\n"
        return voc


class ExtractedFeatures(object):

    def __init__(self, feature_store, extracted_features=None):
        self.feature_store = feature_store
        if extracted_features:
            self.extracted_features = extracted_features.extracted_features
        else:
            self.extracted_features = set(self.feature_store.features.keys())

    def add_feature(self, tpl):
        k,v = tpl
        if k in self.feature_store.lookup:
            valid = self.feature_store.lookup[k][v].union(self.feature_store.lookup[k][None])
            self.extracted_features = self.extracted_features.intersection(valid)

    def add_exclusive_feature(self, tpl):
        k,v = tpl
        if k in self.feature_store.lookup:
            valid = self.feature_store.lookup[k][v]
            self.extracted_features = self.extracted_features.intersection(valid)
        else:
            self.extracted_features = set()


    def get_feature_ids(self):
        return frozenset(self.extracted_features)

def load_vecs(file_name):
    vec_ids = dict()
    infile = open(file_name, "r")
    for line in infile:
        els = line.strip().split()
        jmp, cid = els[0].split(".")
        if cid not in vec_ids:
            vec_ids[cid] = dict()
        try:
            jmp = int(jmp)
        except ValueError:
            pass
        vec_ids[cid][jmp] = sorted(map(int, els[1:]))
    infile.close()
    return vec_ids


def load_weights(file_name):
    d_weights = []
    with open(file_name, "r") as infile:
        for line in infile:
            _, w_id, w = line.strip().split()
            d_weights.append(float(w))

    return np.array(d_weights)

def convoc_reader(fname):
    with open(fname, "r") as infile:
        for line in infile:
            t, con, max_I = line.split()
            max_I = int(max_I)
            yield t, con, max_I

