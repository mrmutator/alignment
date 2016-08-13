import argparse
from CorpusReader import AnnotatedCorpusReader
import numpy as np
import random
import re

def anti_vowel(s):
    result = re.sub(r'[AEIOU]', '', s, flags=re.IGNORECASE)
    return result

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
    outfile = open(outfile_name, "w")
    for (e_toks, f_toks, f_heads, pos, rel, hmm_transitions, order, gold_alignment, ibm1_best, e_str, f_str) in corpus:
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
        start_features_single = []

        start_features_single.append("fj=" + str(f_toks[0]))
        start_features_single.append("relj=" + str(rel[0]))
        start_features_single.append("posj=" + str(pos[0]))


        start_vecs = []
        for i in xrange(I_ext):
            f_ids = []
            start_features_single_i = []
            start_features_single_i.append("rsp=" + str(float(i)/I))
            if ibm1_best[i] == 0:
                start_features_single_i.append("bestibm1")
            if i > 0:
                if e_str[i-1] == f_str[0]:
                    start_features_single_i.append("exact_match")
                if anti_vowel(e_str[-1]) == anti_vowel(f_str[0]):
                    start_features_single_i.append("exact_novowels")
                if e_str[i-1][:3] == f_str[0][:3]:
                    start_features_single_i.append("prefix")
                if e_str[i-1][-3:] == f_str[0][-3:]:
                    start_features_single_i.append("suffix")
                if len(e_str[i-1]) < 4 and len(f_str[0]) < 4:
                    start_features_single_i.append("both_short")


            for sf in start_features_single + start_features_single_i:
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

            j_features_single = []

            j_features_single.append("fj=" + str(f_toks[j]))
            j_features_single.append("posj=" + str(pos[j]))
            j_features_single.append("relj=" + str(rel[j]))


            for ip in xrange(I_ext):
                ip_vecs = []
                for i in xrange(I_ext):
                    f_ids = []
                    j_features_single_i = []
                    j_features_pair_i = []
                    j_features_single_i.append("rsp=" + str(float(i) / I))
                    if ibm1_best[i] == j:
                        j_features_single_i.append("bestibm1")
                    if i > 0:
                        if e_str[i - 1] == f_str[j]:
                            j_features_single_i.append("exact_match")
                        if anti_vowel(e_str[-1]) == anti_vowel(f_str[j]):
                            j_features_single_i.append("exact_novowels")
                        if e_str[i - 1][:3] == f_str[j][:3]:
                            j_features_single_i.append("prefix")
                        if e_str[i - 1][-3:] == f_str[j][-3:]:
                            j_features_single_i.append("suffix")
                        if len(e_str[i - 1]) < 4 and len(f_str[j]) < 4:
                            j_features_single_i.append("both_short")
                        if ip > 0:
                            j_features_pair_i.append("jump_width=" + str(abs(i-ip-1)))


                    for jf in j_features_single + j_features_single_i:
                        fid = all_features.add_feature(jf + ",eaj=" + str(e_ext[i]))
                        f_ids.append(fid)
                    for jf in j_features_pair_i:
                        fid = all_features.add_feature(jf + ",eaj=" + str(e_ext[i]) + ",eajp=" + str(e_ext[ip]))
                        f_ids.append(fid)
                    vec_id = all_vectors.add_vector(frozenset(f_ids))
                    ip_vecs.append(vec_id)
                outfile.write(" ".join(map(str, ip_vecs)) + "\n")
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
    extract_features(corpus, args.corpus + ".extracted")



