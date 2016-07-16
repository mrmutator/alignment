from collections import defaultdict
import numpy as np
import itertools
from scipy.sparse import lil_matrix
from scipy.stats import entropy as kldiv
from matplotlib import pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()

class CorpusReader(object):
    def __init__(self, corpus_file, limit=None):
        self.corpus_file = open(corpus_file, "r")
        self.limit = limit
        self.next = self.__iter_sent

    def reset(self):
        self.corpus_file.seek(0)

    def __iter_sent(self):
        self.reset()
        c = 0
        buffer = []
        b = 0
        for line in self.corpus_file:
            if b != -1:
                buffer.append(map(int, line.strip().split()))
            b += 1
            if b == 7:
                yield buffer
                c += 1
                if c == self.limit:
                    break
                b = -1
                buffer = []


    def __iter__(self):
        return self.next()

    def get_length(self):
        c = 0
        for _ in self:
            c += 1
        return c

class CondIndex(object):

    def __init__(self):
        self.i = 0
        self.index_data = dict()
        self.feature_data = defaultdict(lambda: defaultdict(set))


    def add(self, feature_set):
            if feature_set in self.index_data:
                return self.index_data[feature_set]
            else:
                self.i += 1
                self.index_data[feature_set] = self.i
                for f,v in feature_set:
                    self.feature_data[f][v].add(self.i)
                return self.i

    def get_combinations(self, max=np.inf):
        for num_classes in xrange(1, min(max, len(self.feature_data))+1):
            logger.info("Extracting combinations dim="+str(num_classes))
            for combo in itertools.combinations(self.feature_data, num_classes):
                for feature_set in itertools.product(*map(self.feature_data.__getitem__, combo)):
                    f_pairs = zip(combo, feature_set)
                    involved_id_sets = [self.feature_data[k1][k2] for k1, k2 in f_pairs]
                    intersection = set.intersection(*involved_id_sets)
                    if intersection:
                        yield frozenset(f_pairs), intersection


class Statistics(object):

    def __init__(self, corpus, gold_file, gold_order, sure_only=True, pos_voc_file=None, rel_voc_file=None):
        self.pos_voc = dict()
        self.rel_voc = dict()
        self.dir_voc = {-1: "l", 1:  "r", 0: "-"}
        if pos_voc_file:
            self.pos_voc = self.read_cond_voc_file(pos_voc_file)
        if rel_voc_file:
            self.rel_voc = self.read_cond_voc_file(rel_voc_file)
        self.min_i = 0
        self.max_i = 0
        self.gold_aligned = defaultdict(lambda: defaultdict(set))
        self.read_gold_file(gold_file, gold_order, sure_only=sure_only)
        self.feature_voc = CondIndex()

        self.stats = defaultdict(lambda: defaultdict(float))
        self.set_freq = defaultdict(float)

        self.read_corpus(corpus)

        self.array_length = abs(self.min_i) + self.max_i + 1
        self.make_arrays()




    def read_cond_voc_file(self, fname):
        voc = dict()
        voc[None] = "-"
        with open(fname, "r") as infile:
            for line in infile:
                i, lbl = line.strip().split()
                i = int(i)
                voc[i] = lbl
        return voc

    def read_gold_file(self, file_name, order=("e", "f"), sure_only=True):
        with open(file_name, "r") as infile:
            for line in infile:
                els = line.strip().split()
                snt = int(els[0])
                al_type = els[3]
                if order == ("e", "f"):
                    e = els[1]
                    f = els[2]
                elif order == ("f", "e"):
                    e = els[2]
                    f = els[1]
                else:
                    raise Exception("Invalid order type.")
                if not sure_only or al_type == "S":
                    self.gold_aligned[snt-1][int(f)-1].add(int(e)-1)


    def read_corpus(self, corpus_file):
        corpus = CorpusReader(corpus_file)

        for sent_num, (e_toks, f_toks, f_heads, pos, rel, _, order) in enumerate(corpus):
            J = len(f_toks)
            tree_levels = [0]*J
            tree_levels[0] = 0
            for j in xrange(1,len(f_toks)):
                tree_levels[j] = tree_levels[f_heads[j]] + 1
            # reorder to normal order
            order_indices, _ = zip(*sorted(list(enumerate(order)), key=lambda t: t[1]))
            f_toks = map(f_toks.__getitem__, order_indices)
            pos = map(pos.__getitem__, order_indices)
            rel = map(rel.__getitem__, order_indices)
            # dir = map(dir.__getitem__, order_indices)
            tree_levels = map(tree_levels.__getitem__, order_indices)
            f_heads = [order[f_heads[oi]] for oi in order_indices]

            # translate con Ids
            pos = map(lambda p: self.pos_voc.get(p,p), pos)
            rel = map(lambda p: self.rel_voc.get(p,p), rel)
            # dir = map(lambda p: self.dir_voc.get(p,p), dir)
            # there is some faulty dir data in some parses
            # to be safe, recompute unless it's guaranteed to be correct
            # also uncomment dir above!!!
            dir = [np.sign(f_heads[j]-j) for j in xrange(J)]

            children = [set() for _ in xrange(J)]
            for j, h in enumerate(f_heads):
                if j != h:
                    children[h].add(j)

            for j in xrange(J):
                parents_aligned = self.gold_aligned[sent_num][j]
                if not parents_aligned or (not children[j]):
                    continue
                p_par, r_par, d_par = pos[j], rel[j], dir[j]
                par_left_children = sum([1 for c in children[j] if c < j])
                par_right_children = sum([1 for c in children[j] if c > j])
                par_children = par_left_children + par_right_children


                parent_weight = 1.0 / len(parents_aligned)
                for c in children[j]:
                    c_aligned = self.gold_aligned[sent_num][c]

                    p_c, r_c, d_c = pos[c], rel[c], dir[c]
                    c_left_children = sum([1 for cc in children[c] if cc < c])
                    c_right_children = sum([1 for cc in children[c] if cc > c])
                    c_children = c_left_children + c_right_children

                    for i_p in parents_aligned:
                        for i in c_aligned:
                            features = set()
                            features.add(("ppos",p_par))
                            features.add(("prel",r_par))
                            features.add(("pdir",d_par))
                            features.add(("cpos",p_c))
                            features.add(("crel",r_c))
                            features.add(("cdir",d_c))
                            features.add(("l",c - j))
                            features.add(("absl",abs(c - j)))
                            features.add(("I",len(e_toks)))
                            features.add(("J",J))
                            features.add(("ptl",tree_levels[j]))
                            features.add(("ctl",tree_levels[c]))
                            features.add(("plc", par_left_children))
                            features.add(("prc", par_right_children))
                            features.add(("pc", par_children))
                            features.add(("clc", c_left_children))
                            features.add(("crc", c_right_children))
                            features.add(("cc", c_children))


                            set_id = self.feature_voc.add(frozenset(features))
                            rel_dist = i - i_p
                            weight = (1.0 / len(c_aligned)) * parent_weight
                            self.stats[set_id][rel_dist] += weight
                            self.set_freq[set_id] += weight
                            if rel_dist > self.max_i:
                                self.max_i = rel_dist
                            if rel_dist < self.min_i:
                                self.min_i = rel_dist


    def make_arrays(self):
        for set_id in self.stats:
            array = np.zeros(self.array_length)
            for i, j in enumerate(xrange(self.min_i, self.max_i + 1)):
                array[i] = self.stats[set_id][j]
            self.stats[set_id] = array


    def compute_entropy(self, ids):
        dist = np.zeros(self.array_length)
        for i in ids:
            dist += self.stats[i]
        # smoothing:
        sparse = lil_matrix(dist)
        dist += (np.ones(len(dist)) * 0.00000001)
        dist = dist / np.sum(dist)
        return -np.sum(np.multiply(dist, np.log(dist))), sparse

    def make_plot(self, dist, features, fname):
        dist = dist.toarray().flatten()
        dist = dist / float(np.sum(dist))
        x = np.arange(self.min_i, self.max_i+1)
        plt.bar(x, dist)
        plt.title(",".join([fn + "=" + str(fv) for (fn, fv) in features]))
        plt.savefig(fname + ".png")
        plt.close()

    def compute_concentrations(self, dist, conc_num):
        dist = dist.toarray().flatten()
        q = self.array_length / float(conc_num)
        current = q
        first_part = 0
        limit = int(np.ceil(q))
        sm = 0
        sums = []
        while len(sums) < conc_num:
            sm += np.sum(dist[first_part:limit])
            rest = limit - current
            if len(sums) < conc_num - 1:
                rest_sm = dist[limit - 1] * rest
            else:
                rest_sm = 0
            sm -= rest_sm
            sums.append(sm)
            sm = rest_sm
            current = limit + q - rest
            first_part = limit
            limit = int(np.ceil(current))
        return np.array(sums) / np.sum(sums)





if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-gold", required=True)
    arg_parser.add_argument("-output", required=True)
    arg_parser.add_argument("-order", required=True)
    arg_parser.add_argument("-pos_voc", required=False, default="")
    arg_parser.add_argument("-rel_voc", required=False, default="")
    arg_parser.add_argument("-max_combo", required=False, default=3, type=int)
    arg_parser.add_argument("-result_limit", required=False, default=30, type=int)
    arg_parser.add_argument("-plots", required=False, default="", type=str)
    arg_parser.add_argument("-num_range", required=False, default=3, type=int)


    args = arg_parser.parse_args()

    if args.order == "ef":
        order = ("e", "f")
    elif args.order == "fe":
        order = ("f", "e")
    else:
        raise Exception("Invalid order: "+ args.order)

    stat = Statistics(args.corpus, args.gold, ("f", "e"), pos_voc_file=args.pos_voc, rel_voc_file=args.rel_voc, sure_only=False)

    weighted = defaultdict(dict)
    freqs = dict()
    dists = dict()
    concentrations = defaultdict(lambda: defaultdict(dict))


    smoothing = np.ones(stat.array_length) * 0.000000001
    for features, ids in stat.feature_voc.get_combinations(max=args.max_combo):
        num_features = len(features)
        freq = len(ids)
        if freq > 10:
            entropy, dist = stat.compute_entropy(ids)
            freqs[features] = freq
            weighted[num_features][features] = (1.0/entropy) * freq
            dists[features] = dist
            w_conc = stat.compute_concentrations(dist, args.num_range) * freq
            best_m = np.argmax(w_conc)
            concentrations[best_m][num_features][features] = w_conc




    smoothing = np.ones(stat.array_length) * 0.000000001
    logger.info("Total feature combinations tested: " + str(len(freqs)))
    logger.info("Writing output file and plots.")
    selected = dict()
    for n in xrange(1,args.max_combo+1):
        outfile = open(args.output + ".e" + str(n), "w")
        selected_n = 0
        for features in sorted(weighted[n], key= weighted[n].get, reverse=True):
            current_dist = dists[features].toarray().flatten() + smoothing
            for s, s_dist in selected.iteritems():
                kl = kldiv(s_dist, current_dist)
                if kl < 0.5:
                    break
            else:
                selected_n += 1
                selected[features] = current_dist
                outfile.write(" ".join([",".join([fn + "=" + str(fv) for (fn, fv) in features]), str(weighted[n][features]), str(freqs[features])]) + "\n")
                if args.plots:
                    stat.make_plot(dists[features], features, args.plots.rstrip("/") + "/we" + str(n) + "." + str(selected_n))
            if selected_n == args.result_limit:
                break

        outfile.close()

    for m in xrange(args.num_range):
        selected = dict()
        for n in xrange(1, args.max_combo + 1):
            outfile = open(args.output + ".c" + str(m) + "." + str(n), "w")
            selected_n = 0
            for features in sorted(concentrations[m][n], key= lambda x: concentrations[m][n][x][m], reverse=True):
                current_dist = dists[features].toarray().flatten() + smoothing
                for s, s_dist in selected.iteritems():
                    kl = kldiv(s_dist, current_dist)
                    if kl < 0.5:
                        break
                else:
                    selected_n += 1
                    selected[features] = current_dist
                    outfile.write(" ".join(
                        [",".join([fn + "=" + str(fv) for (fn, fv) in features]), str(np.round(concentrations[m][n][features], 2)), str(freqs[features])]) + "\n")
                    if args.plots:
                        stat.make_plot(dists[features], features,
                                       args.plots.rstrip("/") + "/c" + str(m) + "." + str(n) + "." + str(selected_n))
                if selected_n == args.result_limit:
                    break

            outfile.close()




    # for m in xrange(args.num_range):
    #     outfile = open(args.output + ".c" + str(m), "w")
    #     for i, features in enumerate(sorted(concentrations[m], key= lambda x: concentrations[m][x][m], reverse=True)[:args.result_limit]):
    #         outfile.write(" ".join([",".join([fn + "=" + str(fv) for (fn, fv) in features]), str(np.round(concentrations[m][features], 2)), str(freqs[features])]) + "\n")
    #         if args.plots:
    #             stat.make_plot(dists[features], features, args.plots.rstrip("/") + "/c" + str(m) + "-" + str(i))
    #
    #     outfile.close()

# read gold file

# iterate through corpus

# for each parent
  # for each child
  # see if child is aligned, if yes, add statistics