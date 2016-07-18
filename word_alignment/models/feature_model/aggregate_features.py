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
    all_cons = features.ConditionsVoc()
    all_vecs = features.VectorVoc()

    for f in glob.glob(args.dir + "/*.fvoc"):
        # features
        trans_features = dict()
        with open(f, "r") as infile:
            for line in infile:
                els = line.strip().split("\t")
                file_i = int(els[0])
                feats= []
                for feat in els[1:]:
                    k, v = feat.split()
                    feats.append((k, int(v)))
                global_i = all_features.add(frozenset(feats))
                trans_features[file_i] = global_i


        vec_file = re.sub("\.fvoc", ".vecvoc", f)
        outfile = open(vec_file + ".trans", "w")
        with open(vec_file, "r") as infile:
            for line in infile:
                els = line.strip().split("\t")
                file_vec_i = int(els[0])
                true_feature_ids = map(trans_features.__getitem__, map(int, els[1:]))
                global_vec_id = all_vecs.get_id(frozenset(true_feature_ids))
                outfile.write(" ".join(map(str, [file_vec_i, global_vec_id])) + "\n")
        outfile.close()

        con_file = re.sub("\.fvoc", ".convoc", f)
        outfile = open(con_file + ".trans", "w")
        with open(con_file, "r") as infile:
            for line in infile:
                els = line.strip().split("\t")
                file_con_i = int(els[0])
                con_numbers = map(int, els[1:])
                global_con_id = all_cons.get_id(frozenset(con_numbers))
                outfile.write(" ".join(map(str, [file_con_i, global_con_id])) + "\n")
        outfile.close()


        # annotated_file = re.sub("\.fvoc", ".extracted.gz", f)

        # outfile = gzip.open(annotated_file + ".corrected.gz", "wb")
        # with gzip.open(annotated_file, "rb") as infile:
        #     c = 0
        #     for line in infile:
        #         c += 1
        #         if line.strip() == "":
        #             c = 0
        #         if c > 3:
        #             line = " ".join(map(str, map(trans_cons.__getitem__, map(int, line.strip().split())))) + "\n"
        #         outfile.write(line)
        # outfile.close()

    prefix = re.search("^\./(.*?)\.\d+\.sub_feat", f).group(1)
    with open(prefix + ".features", "w") as outfile:
        outfile.write(all_features.get_voc())

    with open(prefix + ".vecs", "w") as outfile:
        outfile.write(all_vecs.get_voc())

    with open(prefix + ".cons", "w") as outfile:
        outfile.write(all_cons.get_voc())

    with open(prefix + ".weights", "w") as outfile:
        for w_id in sorted(all_features.feature_dict.values()):
            w = random_weight()
            outfile.write("w " + str(w_id) + " " + str(w) + "\n")
