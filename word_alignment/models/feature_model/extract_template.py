import argparse
from CorpusReader import CorpusReader
import gzip
import numpy as np
from collections import defaultdict

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


def extract_features(corpus, feature_pool, out_file_name):
    all_jmp_condition_ids = defaultdict(max_dict)
    all_start_condition_ids = defaultdict(max_dict)
    outfile = gzip.open(out_file_name + ".extracted.gz", "w")
    for e_toks, f_toks, f_heads, pos, rel, hmm_transitions, order in corpus:


        outfile.write(" ".join(map(str, e_toks)) + "\n")
        outfile.write(" ".join(map(str, f_toks)) + "\n")
        outfile.write(" ".join(map(str, f_heads)) + "\n")

        J = len(f_toks)
        I = len(e_toks)
        tree_levels = [0] * J

        dir = [np.sign(order[j] - order[f_heads[j]]) for j in xrange(J)]

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
        features_sentence_level = ExtractedFeatures(feature_pool)
        features_sentence_level.add_feature(("I", I))
        features_sentence_level.add_feature(("J", J))


        # j=0
        j=0
        features_0 = ExtractedFeatures(feature_pool, features_sentence_level)
        j_tree_level = 0
        features_0.add_feature(("cpos", pos[j]))
        features_0.add_feature(("crel", rel[j]))
        features_0.add_feature(("cdir", dir[j]))
        features_0.add_feature(("ctl", j_tree_level))
        features_0.add_feature(("clc", left_children[j]))
        features_0.add_feature(("crc", right_children[j]))
        features_0.add_feature(("cc", children[j]))
        features_0.add_exclusive_feature(("j", j))
        features_0.add_feature(("oj", order[j]))

        conditions = features_0.get_feature_ids()
        if not conditions:
            conditions = frozenset([0])
        condition_id = "-".join(map(str, sorted(conditions)))
        all_start_condition_ids[condition_id].add(I)

        # do all the writing here
        outfile.write(condition_id + "\n")


        # rest
        for j in xrange(1, J):
            features_j = ExtractedFeatures(feature_pool, features_sentence_level)
            # add features
            h = f_heads[j]
            j_tree_level = tree_levels[h] + 1
            tree_levels[j] = j_tree_level
            features_j.add_feature(("ppos", pos[h]))
            features_j.add_feature(("prel", rel[h]))
            features_j.add_feature(("pdir", dir[h]))
            features_j.add_feature(("cpos", pos[j]))
            features_j.add_feature(("crel", rel[j]))
            features_j.add_feature(("cdir", dir[j]))
            features_j.add_feature(("l", order[j]-order[h]))
            features_j.add_feature(("absl", abs(order[j]-order[h])))
            features_j.add_feature(("ctl", j_tree_level))
            features_j.add_feature(("ptl", tree_levels[j]))
            features_j.add_feature(("plc", left_children[h]))
            features_j.add_feature(("prc", right_children[h]))
            features_j.add_feature(("pc", children[h]))
            features_j.add_feature(("clc", left_children[j]))
            features_j.add_feature(("crc", right_children[j]))
            features_j.add_feature(("cc", children[j]))
            features_j.add_feature(("j", j))
            features_j.add_feature(("pj", h))
            features_j.add_feature(("oj", order[j]))
            features_j.add_feature(("op", order[h]))
            features_j.add_feature(("phmm", hmm_transitions[h]))
            features_j.add_feature(("chmm", hmm_transitions[j]))


            for i_p in xrange(I):
                features_i_p = ExtractedFeatures(feature_pool, features_j)
                # add features
                features_i_p.add_feature(("ip", i_p))

                # static features complete
                # check if there is a feature that matches
                conditions = features_i_p.get_feature_ids()
                if not conditions:
                    conditions = frozenset([0])
                condition_id = "-".join(map(str, sorted(conditions)))
                all_jmp_condition_ids[condition_id].add(I)

                # do all the wirting here
                outfile.write(condition_id + "\n")

        outfile.write("\n")

    outfile.close()


    with open(out_file_name + ".convoc", "w") as outfile:
        for cid in all_start_condition_ids:
            max_I = all_start_condition_ids[cid].get()
            outfile.write(" ".join(["s", cid, str(max_I)]) + "\n")

        for cid in all_jmp_condition_ids:
            max_I = all_jmp_condition_ids[cid].get()
            outfile.write(" ".join(["j", cid, str(max_I)]) + "\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-feature_file", required=True)
    args = arg_parser.parse_args()
    corpus = CorpusReader(args.corpus)

    feature_pool = FeatureStore(args.feature_file)

    extract_features(corpus, feature_pool, args.corpus)