import glob
import argparse
from features import Features
import random
from collections import defaultdict
import re

def random_weight():
    return random.uniform(-1, 1)


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


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-dir", required=True)
    args = arg_parser.parse_args()

    all_cons = defaultdict(lambda: defaultdict(max_dict))
    all_features = Features()

    for f in glob.glob(args.dir + "/*.convoc"):
        with open(f, "r") as infile:
            for line in infile:
                ftype, con_id, max_I = line.split()
                all_cons[ftype][con_id].add(int(max_I))

    prefix = re.search("^\./(.*?)\.\d+\.sub_feat", f).group(1)
    outfile = open(prefix + ".vecs", "w")

    for con_id in all_cons["s"]:
        max_I = all_cons["s"][con_id].get()
        ftype_ids = map(int, con_id.split("-"))
        for jmp in xrange(max_I):
            feature_ids = set()
            for ftype in ftype_ids:
                f_tuple = (ftype, jmp)
                feature_id = all_features.add(f_tuple)
                feature_ids.add(feature_id)
            vec_id = ".".join([str(jmp), con_id])
            outfile.write(" ".join([vec_id] + map(str, feature_ids)) + "\n")

    for con_id in all_cons["j"]:
        max_I = all_cons["j"][con_id].get()
        ftype_ids = map(int, con_id.split("-"))
        for jmp in xrange(-max_I+1, max_I):
            feature_ids = set()
            for ftype in ftype_ids:
                f_tuple = (ftype, jmp)
                feature_id = all_features.add(f_tuple)
                feature_ids.add(feature_id)
            vec_id = ".".join([str(jmp), con_id])
            outfile.write(" ".join([vec_id] + map(str, feature_ids)) + "\n")

    outfile.close()

    with open(prefix + ".weights", "w") as outfile:
        for w_id in sorted(all_features.feature_dict.values()):
            w = random_weight()
            outfile.write("w " + str(w_id) + " " + str(w) + "\n")

    with open(prefix + ".fvoc", "w") as outfile:
        outfile.write(all_features.get_voc())
