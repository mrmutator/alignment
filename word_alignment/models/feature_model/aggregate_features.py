import glob
import argparse
from features import Features
import random
from collections import defaultdict
import re
from features import max_dict
import numpy as np

def random_weight():
    return random.uniform(-0.2, 0.2)

def uniform_weight():
    return 0.5


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-dir", required=True)
    arg_parser.add_argument("-uniform", action="store_true", default=False)
    arg_parser.add_argument("-max_jump", type=int, default=0)
    args = arg_parser.parse_args()

    if args.max_jump == 0:
        max_pos_jmp = np.inf
        max_neg_jmp = -np.inf
    else:
        max_pos_jmp = args.max_jump
        max_neg_jmp = -args.max_jump

    if args.uniform:
        assign_weight = uniform_weight
    else:
        assign_weight = random_weight

    all_cons = defaultdict(lambda: defaultdict(max_dict))
    all_features = Features()

    for f in glob.glob(args.dir + "/*.convoc"):
        with open(f, "r") as infile:
            for line in infile:
                ftype, con_id, max_I = line.split()
                all_cons[ftype][con_id].add(int(max_I))

    prefix = re.search("^\./(.*?)\.\d+\.sub_feat", f).group(1)
    #outfile = open(prefix + ".vecs", "w")
    convoc_outfile = open(prefix + ".convoc_list", "w")

    for con_id in all_cons["s"]:
        max_I = all_cons["s"][con_id].get()
        #convoc_outfile.write(" ".join(["s", con_id, str(max_I)]) + "\n")
        ftype_ids = map(int, con_id.split("-"))
        for jmp in xrange(max_I):   
            feature_ids = set()
            for ftype in ftype_ids:
                f_tuple = (ftype, jmp)
                feature_id = all_features.add(f_tuple)
                feature_ids.add(feature_id)
            #vec_id = ".".join([str(jmp), con_id])
            convoc_outfile.write(" ".join(["s", con_id, str(jmp)] + map(str, feature_ids)) + "\n")

    for con_id in all_cons["j"]:
        max_I = all_cons["j"][con_id].get()
        #convoc_outfile.write(" ".join(["j", con_id, str(max_I)]) + "\n")
        ftype_ids = map(int, con_id.split("-"))
        for jmp in xrange(-max_I+1, max_I):
            feature_ids = set()
            for ftype in ftype_ids:
                if jmp > 0:
                    act_jmp = min(max_pos_jmp, jmp)
                elif jmp < 0:
                    act_jmp = max(max_neg_jmp, jmp)
                else:
                    act_jmp = jmp
                f_tuple = (ftype, act_jmp)
                feature_id = all_features.add(f_tuple)
                feature_ids.add(feature_id)
            #vec_id = ".".join([str(jmp), con_id])
            #outfile.write(" ".join([vec_id] + map(str, feature_ids)) + "\n")
            convoc_outfile.write(" ".join(["j", con_id, str(jmp)] + map(str, feature_ids)) + "\n")

    #outfile.close()
    convoc_outfile.close()

    with open(prefix + ".weights", "w") as outfile:
        for w_id in sorted(all_features.feature_dict.values()):
            w = assign_weight()
            outfile.write("w " + str(w_id) + " " + str(w) + "\n")

    with open(prefix + ".fvoc", "w") as outfile:
        outfile.write(all_features.get_voc())
