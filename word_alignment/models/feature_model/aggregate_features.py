import gzip
import glob
import argparse
import features
import re
import random

def random_weight():
    return random.uniform(-1, 1)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-dir", required=True)
    args = arg_parser.parse_args()

    all_features = features.Features()
    all_cons = features.FeatureConditions()

    for f in glob.glob(args.dir + "/*.fvoc"):
        trans_features = dict()
        with open(f, "r") as infile:
            for line in infile:
                file_i, feat = line.strip().split()
                global_i = all_features.add(feat)
                trans_features[int(file_i)] = global_i

        con_file = re.sub("\.fvoc", ".convoc", f)
        trans_cons = dict()
        #outfile = open(con_file + ".corrected", "w")
        with open(con_file, "r") as infile:
            for line in infile:
                try:
                    els = map(int, line.strip().split())
                    file_con_i = els[0]
                    true_feature_ids = map(trans_features.__getitem__, els[1:])
                    true_con_id = all_cons.get_id(frozenset(true_feature_ids))
                    trans_cons[file_con_i] = true_con_id
                except ValueError:
                    file_con_i, ftuple = line.strip().split("\t")
                    ftuple = tuple(ftuple.split())
                    true_condition_id = all_cons.get_id(ftuple)
                    trans_cons[int(file_con_i)] = true_condition_id

                #outfile.write(" ".join(map(str, [true_con_id] + list(true_feature_ids))) + "\n")

        #outfile.close()

        annotated_file = re.sub("\.fvoc", ".extracted.gz", f)

        outfile = gzip.open(annotated_file + ".corrected.gz", "wb")
        with gzip.open(annotated_file, "rb") as infile:
            c = 0
            for line in infile:
                c += 1
                if line.strip() == "":
                    c = 0
                if c > 3:
                    line = " ".join(map(str, map(trans_cons.__getitem__, map(int, line.strip().split())))) + "\n"
                outfile.write(line)
        outfile.close()

    prefix = re.search("^\./(.*?)\.\d+\.sub_feat",f).group(1)
    with open(prefix + ".features", "w") as outfile:
        outfile.write(all_features.get_voc())

    with open(prefix + ".cons", "w") as outfile:
        outfile.write(all_cons.get_featureset_voc())

    with open(prefix + ".static", "w") as outfile:
        outfile.write(all_cons.get_condition_voc())

    with open(prefix + ".weights", "w") as outfile:
        for w_id in sorted(all_features.feature_dict.values()):
            w = random_weight()
            outfile.write("w " + str(w_id) + " " + str(w) + "\n")

