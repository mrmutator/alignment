import argparse
from CorpusReader import AnnotatedCorpusReader
import numpy as np
import random

def random_weight():
    return random.uniform(-1, 1)

class Features(object):

    def __init__(self):
        self.i = 0
        self.fdict = dict()

    def add_feature(self, feature):
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

def extract_features(corpus, outfile_name):
    all_features = Features()
    all_vectors = Vectors()
    outfile = open(outfile_name + ".extracted", "w")
    for (e_toks, f_toks, f_heads, pos, rel, hmm_transitions, order, gold_alignment) in corpus:
        J = len(f_toks)
        I = len(e_toks)
        outfile.write("\n".join([str(I), " ".join(map(str, f_heads)), " ".join(map(str, gold_alignment)), ""]))


        tree_levels = [0] * J

        I_ext = I+1
        e_ext = ["NULL"] + e_toks

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

        # start
        start_features = []

        start_features.append("fj=" + str(f_toks[0]))


        start_vecs = []
        for i in xrange(I_ext):
            f_ids = []
            for sf in start_features:
                fid = all_features.add_feature(sf + ",eaj=" + str(e_ext[i]))
                f_ids.append(fid)
            vec_id = all_vectors.add_vector(frozenset(f_ids))
            start_vecs.append(vec_id)
        outfile.write(" ".join(map(str, start_vecs)) + "\n")


        # rest
        for j in xrange(1, J):

            h = f_heads[j]
            j_tree_level = tree_levels[h] + 1
            tree_levels[j] = j_tree_level

            j_features = []
            j_features.append("fj=" + str(f_toks[0]))
            j_features.append("posj=" + str(pos[0]))

            for ip in xrange(I_ext):
                ip_vecs = []
                for i in xrange(I_ext):
                    f_ids = []
                    for jf in j_features:
                        fid = all_features.add_feature(jf + ",eaj=" + str(e_ext[i]) + ",eajp=" + str(e_ext[ip]))
                        f_ids.append(fid)
                    vec_id = all_vectors.add_vector(frozenset(f_ids))
                    ip_vecs.append(vec_id)
                outfile.write(" ".join(map(str, start_vecs)) + "\n")
        outfile.write("\n")

    outfile.close()
    with open(outfile_name + ".vecs", "w") as outfile:
        for s in all_vectors.get_voc():
            outfile.write(s)

    with open(outfile_name + ".fvoc", "w") as outfile:
        for s in all_features.get_voc():
            outfile.write(s)

    with open(outfile_name + ".weights", "w") as outfile:
        for i, s in enumerate(all_features.generate_weights()):
            outfile.write(" ".join(map(str, [i,s])) + "\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    args = arg_parser.parse_args()
    corpus = AnnotatedCorpusReader(args.corpus)
    extract_features(corpus, "corpus")



