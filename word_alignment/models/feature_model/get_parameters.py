from collections import defaultdict
import random
import numpy as np
import argparse
from CorpusReader import CorpusReader

def read_reorder_file(f):
    if not f:
        return None
    reorder_dict = dict()
    with open(f, "r") as infile:
        for line in infile:
            pos, left, right = map(int, line.strip().split())
            reorder_dict[pos] = (left, right)
    return reorder_dict


def reorder(data, order):
    """
    Order is a list with same length as data that specifies for each position of data, which rank it has in the new order.
    :param data:
    :param order:
    :return:
    """
    new_data = [None] * len(data)
    for i, j in enumerate(order):
        new_data[j] = data[i]
    return new_data


def hmm_reorder(f_toks, pos, rel, dir, order):
    # HMM reorder
    J = len(f_toks)
    new_f_toks = reorder(f_toks, order)
    new_pos = reorder(pos, order)
    new_rel = reorder(rel, order)
    new_dir = reorder(dir, order)
    new_f_heads = [0] + range(J - 1)
    new_order = range(J)
    return new_f_toks, new_f_heads, new_pos, new_rel, new_dir, new_order


def random_prob():
    return random.random() * -1 + 1  # random number between 0 and 1, excluding 0, including 1


class Features(object):

    # static and dynamic features
    # static ones are stored in subcorpus file so they don't need to be extracted again
    # dynamic ones need to be accounted for (reserve a spot in feature vector) and generate weights

    def __init__(self):
        self.i = 0
        self.feature_dict = dict()

    def add(self, feat):
        if feat not in self.feature_dict:
            self.feature_dict[feat] = self.i
            self.i += 1
        return self.feature_dict[feat]


    def extract_features(self):
        tree_level = [0]
        # externalize this
        for j in range(1, len(f_toks)):
            par = f_heads[j]
            tree_level.append(tree_level[par] + 1)
            orig_tok_pos = order[j]
            orig_head_pos = order[par]
            parent_distance = abs(orig_head_pos - orig_tok_pos)
            # pos[par], rel[par], dir[par], pos[j], rel[j], dir[j], parent_distance, tree_level[j]
        return


class Parameters(object):
    def __init__(self, corpus, p_0=0.2):
        self.corpus = corpus
        self.cooc = set()
        self.lengths = set()
        self.p_0 = p_0

        self.features = Features()

        self.c = 0
        self.add_corpus(corpus)

        self.t_params = dict()
        self.s_params = dict()

    def add_corpus(self, corpus):
        self.c = 0
        for e_toks, f_toks, _, _, _, _, _ in corpus:
            self.c += 1
            I = len(e_toks)
            self.lengths.add(I)
            for f in f_toks:
                self.cooc.add((0, f))
                for e in e_toks:
                    self.cooc.add((e, f))

    def initialize_trans_t_file(self, t_file):
        trans_dict = defaultdict(dict)
        with open(t_file, "r") as infile:
            for line in infile:
                e, f, p = line.strip().split()
                e = int(e)
                f = int(f)
                if (e, f) in self.cooc:
                    trans_dict[e][f] = float(p)
        for e in trans_dict:
            Z = np.sum(trans_dict[e].values())
            for f in trans_dict[e]:
                self.t_params[(e, f)] = trans_dict[e][f] / float(Z)
        del self.cooc

    def write_params(self, max_I, sub_t, out_file_name):
        outfile = open(out_file_name, "w")
        for key in sub_t:
            value = self.t_params[key]
            key_str = ["t"] + map(str, [key[0], key[1], value])
            outfile.write(" ".join(key_str) + "\n")

        for i in range(max_I)
            for i in xrange(I):
                value = self.s_params[(I, i)]
                key_str = ["s"] + map(str, [I, i, value])
                outfile.write(" ".join(key_str) + "\n")


        outfile.close()

    def split_data_get_parameters(self, corpus, file_prefix, num_sentences, hmm=False):
        subset_id = 1
        outfile_corpus = open(file_prefix + ".corpus." + str(subset_id), "w")
        sub_t = set()
        sub_I = set()
        subset_c = 0
        total = 0
        for e_toks, f_toks, f_heads, pos, rel, dir, order in corpus:
            subset_c += 1
            total += 1
            if hmm:
                f_toks, f_heads, pos, rel, dir, order = hmm_reorder(f_toks, pos, rel, dir, order)

            # feature extraction

            # start tok j=0?

            # extract features and return for each position a list of feature_ids
            feature_ids = self.features.extract_features(e_toks, f_toks, f_heads, pos, rel, dir, order)

            # produce subcorpus file
            outfile_corpus.write(" ".join([str(w) for w in e_toks]) + "\n")
            outfile_corpus.write(" ".join([str(w) for w in f_toks]) + "\n")
            outfile_corpus.write(" ".join([str(h) for h in f_heads]) + "\n")
            outfile_corpus.write(" ".join([str(o) for o in order]) + "\n\n")
            for i, f_ids in enumerate(feature_ids):
                outfile_corpus.write(str(i) + "\t" + " ".join([f_ids]) + "\n")

            for e in e_toks + [0]:
                for f in f_toks:
                    if (e, f) in self.t_params:
                        sub_t.add((e, f))

            if subset_c == num_sentences:
                outfile_corpus.close()
                self.write_params(max(sub_I), sub_t, file_prefix + ".params." + str(subset_id))
                if total < self.c:
                    subset_id += 1
                    outfile_corpus = open(file_prefix + ".corpus." + str(subset_id), "w")
                    sub_t = set()
                    sub_I = set()
                    subset_c = 0
        if subset_c > 0:
            outfile_corpus.close()
            self.write_params(max(sub_I), sub_t, file_prefix + ".params." + str(subset_id))


def prepare_data(corpus, t_file, num_sentences, p_0=0.2, file_prefix="", init_c=1.0, init_t=1.0, tj_cond_head="", tj_cond_tok="",
                 cj_cond_head="", cj_con_tok="", hmm=False, mixed_model={}):
    parameters = Parameters(corpus, p_0=p_0, init_c=init_c, init_t=init_t)


    parameters.initialize_trans_t_file(t_file)

    parameters.split_data_get_parameters(corpus, file_prefix, num_sentences, tj_head_con=tj_cond_head,
                                         tj_tok_con=tj_cond_tok, cj_head_con=cj_cond_head, cj_tok_con=cj_con_tok,
                                         hmm=hmm, mixed_model=mixed_model)
    with open(file_prefix + ".condvoc", "w") as outfile:
        outfile.write(parameters.cond_voc.get_voc())


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-output_prefix", required=True)
    arg_parser.add_argument("-t_file", required=True)
    arg_parser.add_argument("-limit", required=False, type=int, default=0)
    arg_parser.add_argument("-group_size", required=False, type=int, default=-1)
    arg_parser.add_argument("-p_0", required=False, default=0.2, type=float)
    arg_parser.add_argument('-hmm', dest='hmm', action='store_true', default=False)

    args = arg_parser.parse_args()


    corpus = CorpusReader(args.corpus, limit=args.limit)

    prepare_data(corpus=corpus, t_file=args.t_file, num_sentences=args.group_size, p_0=args.p_0,
                 file_prefix=args.output_prefix, init_c = args.init_c, init_t=args.init_t, tj_cond_head=args.tj_cond_head,
                 tj_cond_tok=args.tj_cond_tok,
                 cj_con_tok=args.cj_cond_tok, cj_cond_head=args.cj_cond_head, hmm=args.hmm,
                 mixed_model=read_reorder_file(args.mixed))
