import argparse
from CorpusReader import CorpusReader
import features as fm
import gzip
import numpy as np

def read_feature_file(file_name):
    features = set()
    with open(file_name, "r") as infile:
        for line in infile:
            if line.strip():
                f_pairs = line.strip().split("\t")
                f_pairs = [tuple(p.split(" ")) for p in f_pairs]
                subfeatures = frozenset([(k, int(v)) for k,v in f_pairs])
                features.add(subfeatures)

    return features


def extract_features(corpus, feature_pool, out_file_name):
    feature_voc = fm.Features()
    vector_ids = fm.FeatureConditions()
    outfile = gzip.open(out_file_name + ".extracted.gz", "w")
    for e_toks, f_toks, f_heads, pos, rel, _, order in corpus:
        # because of dir bug in parsing code
        # not all corpora have been updated.
        # manually compute dir
        dir = [0] + [np.sign(order[f_heads[j]] - order[j]) for j in xrange(1, len(f_toks))]

        outfile.write(" ".join(map(str, e_toks)) + "\n")
        outfile.write(" ".join(map(str, f_toks)) + "\n")
        outfile.write(" ".join(map(str, f_heads)) + "\n")

        J = len(f_toks)
        I = len(e_toks)
        tree_levels = [0] * J

        children = [0] * J
        left_children = [0] * J
        right_children = [0] * J

        for j, h in enumerate(xrange(1,J)):
            children[h] += 1
            if order[j] < order[h]:
                left_children[h] += 1
            else:
                right_children[h] += 1


        # sentence_level
        features_sentence_level = set()
        features_sentence_level.add(("I", I))
        features_sentence_level.add(("J", J))


        # j=0



        # rest
        for j in xrange(1, J):
            features_j = set(features_sentence_level)
            # add features
            h = f_heads[j]
            j_tree_level = tree_levels[h] + 1
            tree_levels[j] = j_tree_level
            features_j.add(("ppos", pos[h]))
            features_j.add(("prel", rel[h]))
            features_j.add(("pdir", dir[h]))
            features_j.add(("cpos", pos[j]))
            features_j.add(("crel", rel[j]))
            features_j.add(("cdir", dir[j]))
            features_j.add(("l", order[j]-order[h]))
            features_j.add(("absl", abs(order[j]-order[h])))
            features_j.add(("ctl", j_tree_level))
            features_j.add(("ptl", tree_levels[j]))
            features_j.add(("plc", left_children[h]))
            features_j.add(("prc", right_children[h]))
            features_j.add(("pc", children[h]))
            features_j.add(("clc", left_children[j]))
            features_j.add(("crc", right_children[j]))
            features_j.add(("cc", children[j]))


            for i_p in xrange(I):
                features_i_p = set(features_j)
                # add featuers

                # static features complete
                # check if there is a feature that matches
                conditions = []
                for cand in feature_pool:
                    if cand.issubset(features_i_p):
                        conditions.append(cand)
                if not conditions:
                    conditions = [frozenset([("empty", 0)])]
                condition_id = vector_ids.get_id(tuple(conditions))
                vectors = []
                for i in xrange(I):
                    features_i = set()
                    # add dynamic features
                    for cond in conditions:
                        cond_set = set(cond)
                        cond_set.add(("jmp", i-i_p))
                        feature_id = feature_voc.add(frozenset(cond_set))
                        features_i.add(feature_id)
                    vector_id = vector_ids.get_id(frozenset(features_i))
                    vectors.append(vector_id)


                # do all the wirting here
                outfile.write(" ".join(map(str, [condition_id] + vectors)) + "\n")


        outfile.write("\n")

    outfile.close()

    with open(out_file_name + ".fvoc", "w") as outfile:
        outfile.write(feature_voc.get_voc())

    with open(out_file_name + ".convoc", "w") as outfile:
        outfile.write(vector_ids.get_voc())


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-feature_file", required=True)
    args = arg_parser.parse_args()
    corpus = CorpusReader(args.corpus)

    feature_pool = read_feature_file(args.feature_file)

    extract_features(corpus, feature_pool, args.corpus)
