import argparse
from CorpusReader import AnnotatedCorpusReader
import numpy as np
import re
from features import Features, load_weights
from scipy.sparse import lil_matrix

def viterbi(J, I_ext, f_heads, start_prob, d_probs, order):
    f = np.zeros((J, I_ext))
    f_in = np.zeros((J, I_ext))
    best = np.zeros((J, I_ext), dtype=int)

    for j in reversed(xrange(1, J)):
        values = (np.log(d_probs[j - 1])) + f_in[j]
        best_is = np.argmax(values, axis=1)
        best[j] = best_is
        f[j] = values[np.arange(I_ext), best_is]
        f_in[f_heads[j]] += f[j]

    f[0] = np.log(start_prob) + f_in[0]

    last_best = np.argmax(f[0])
    alignment = [int(last_best)]
    for j in range(1, J):
        head = f_heads[j]
        if head == 0:
            alignment.append(best[j][last_best])
        else:
            alignment.append(best[j][alignment[head]])

    # alignment = [(al, order[j]) for j, al in enumerate(alignment) if al < I]
    alignment = [str(al - 1) + "-" + str(order[j]) for j, al in enumerate(alignment) if al > 0]
    return alignment



def anti_vowel(s):
    result = re.sub(r'[AEIOU]', '', s, flags=re.IGNORECASE)
    return result

def extract_features(corpus, outfile_name, fvoc, weights):
    feature_dim = len(weights)
    all_features = Features(fname=fvoc)
    outfile = open(outfile_name, "w")
    for (e_toks, f_toks, f_heads, pos, rel, hmm_transitions, order, gold_alignment, ibm1_best, e_str, f_str) in corpus:
        J = len(f_toks)
        I = len(e_toks)


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
        #start_features_single.append("relj=" + str(rel[0]))
        #start_features_single.append("posj=" + str(pos[0]))
        #start_features_single.append("start")


        feature_matrix = lil_matrix((I_ext, feature_dim))
        for i in xrange(I_ext):
            f_ids = [0]
            start_features_single_i = []
            #start_features_single_i.append("rsp=" + str(float(i)/I))
            if ibm1_best[i] == 0:
                start_features_single_i.append("bestibm1")
            if i > 0:
                start_features_single_i.append("diag=" + str(i-order[0]))
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


            for sf in start_features_single:
                fid = all_features.add_feature(sf + ",eaj=" + str(e_ext[i]))
                #fid = all_features.add_feature(sf + ",sj=" + str(i if i > 0 else "NULL"))
                if fid is not None:
                    f_ids.append(fid)
            for sf in start_features_single_i:
                # fid = all_features.add_feature(sf + ",eaj=" + str(e_ext[i]))
                fid = all_features.add_feature(sf)
                if fid is not None:
                    f_ids.append(fid)


            feature_matrix.rows[i] = f_ids
            feature_matrix.data[i] = [1.0] * len(f_ids)

        start_feature_matrix = feature_matrix.tocsr()
        start_prob = np.exp(start_feature_matrix.dot(weights))

        # rest
        d_probs = np.zeros((J - 1, I_ext, I_ext))
        for j in xrange(1, J):

            h = f_heads[j]
            j_tree_level = tree_levels[h] + 1
            tree_levels[j] = j_tree_level

            j_features_single = []

            j_features_single.append("fj=" + str(f_toks[j]))
            #j_features_single.append("posj=" + str(pos[j]))
            #j_features_single.append("relj=" + str(rel[j]))




            for ip in xrange(I_ext):
                feature_matrix = lil_matrix((I_ext, feature_dim))
                for i in xrange(I_ext):
                    f_ids = [0]
                    j_features_single_i = []
                    j_features_pair_i = []
                    #j_features_single_i.append("rsp=" + str(round(float(i) / I, 1)))
                    if ibm1_best[i] == j:
                        j_features_single_i.append("bestibm1")

                    if i > 0:
                        if ip > 0:
                            j_features_single_i.append("srclen=" + str(order[j] - order[f_heads[j]]) + ",jmp=" + str(i - ip))
                            j_features_single_i.append("crel=" + str(rel[j]) + ",prel=" + str(rel[f_heads[j]]) + ",jmp=" + str(i-ip))
                        j_features_single_i.append("diag=" + str(i - order[j]))
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
                            j_features_pair_i.append("jump_width=" + str(i - ip - 1))


                    for jf in j_features_single:
                        fid = all_features.add_feature(jf + ",eaj=" + str(e_ext[i]))
                        #fid = all_features.add_feature(jf + ",jmp=" + str(jmp))
                        if fid is not None:
                            f_ids.append(fid)
                    for jf in j_features_pair_i + j_features_single_i:
                        # fid = all_features.add_feature(jf + ",eaj=" + str(e_ext[i]) + ",eajp=" + str(e_ext[ip]))
                        fid = all_features.add_feature(jf)
                        if fid is not None:
                            f_ids.append(fid)

                    # do something with f_ids
                    feature_matrix.rows[i] = f_ids
                    feature_matrix.data[i] = [1.0] * len(f_ids)

                ip_feature_matrix = feature_matrix.tocsr()
                d_probs[j - 1, ip, :I + 1] = np.exp(ip_feature_matrix.dot(weights))

        alignment = viterbi(J, I_ext, f_heads, start_prob, d_probs, order)
        outfile.write(" ".join(alignment) + "\n")

    outfile.close()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-fvoc", required=False, default="")
    arg_parser.add_argument("-weights", required=False, default="")
    args = arg_parser.parse_args()
    corpus = AnnotatedCorpusReader(args.corpus)
    extract_features(corpus, args.corpus + ".aligned", args.fvoc, load_weights(args.weights))



