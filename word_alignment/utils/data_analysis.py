from collections import defaultdict
import numpy as np
import itertools

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
        self.data = dict()
        self.pos_p = defaultdict(set)
        self.rel_p = defaultdict(set)
        self.dir_p = defaultdict(set)
        self.pos_c = defaultdict(set)
        self.rel_c = defaultdict(set)
        self.dir_c = defaultdict(set)


    def add(self, pos_p, rel_p, dir_p, pos_c, rel_c, dir_c):
            if (pos_p, rel_p, dir_p, pos_c, rel_c, dir_c) in self.data:
                return self.data[(pos_p, rel_p, dir_p, pos_c, rel_c, dir_c)]
            else:
                self.i += 1
                self.data[(pos_p, rel_p, dir_p, pos_c, rel_c, dir_c)] = self.i
                self.pos_p[pos_p].add(self.i)
                self.rel_p[rel_p].add(self.i)
                self.dir_p[dir_p].add(self.i)
                self.pos_c[pos_c].add(self.i)
                self.rel_c[rel_c].add(self.i)
                self.dir_c[dir_c].add(self.i)
                return self.i

    def get_dist(self, parent="", tok=""):
        dist_keys = []
        conditions = [x in parent for x in "prd"] + [x in tok for x in  "prd"]
        targets = [self.pos_p, self.rel_p, self.dir_p, self.pos_c, self.rel_c, self.dir_c]
        for i, c in enumerate(conditions):
            if c:
                dist_keys.append(targets[i].keys())
            else:
                dist_keys.append([None])

        parent_keys = dist_keys[:3]
        child_keys = dist_keys[3:]

        parent_combinations = itertools.product(*parent_keys)
        child_combinations = itertools.product(*child_keys)

        parent_dists = dict()
        for p, r, d in parent_combinations:
            all = set(range(self.i))
            if p is not None:
                all = all.intersection(self.pos_p[p])
            if r is not None:
                all = all.intersection(self.rel_p[r])
            if d is not None:
                all = all.intersection(self.dir_p[d])
            parent_dists[p, r, d] = all

        child_dists = dict()
        for p, r, d in child_combinations:
            all = set(range(self.i))
            if p is not None:
                all = all.intersection(self.pos_c[p])
            if r is not None:
                all = all.intersection(self.rel_c[r])
            if d is not None:
                all = all.intersection(self.dir_c[d])
            child_dists[p, r, d] = all

        all = dict()
        for (pp, pr, pd), p_set in parent_dists.iteritems():
            for (cp, cr, cd), c_set in child_dists.iteritems():
                total = p_set.intersection(c_set)
                if total:
                    all[pp, pr, pd, cp, cr, cd] = total
        return all


    def get_pos_p_dist(self):
        for pos in self.pos_p:
            yield pos, self.pos_p[pos]

    def get_rel_p_dist(self):
        for rel in self.rel_p:
            yield rel, self.rel_p[rel]

    def get_dir_p_dist(self):
        for dir in self.dir_p:
            yield dir, self.dir_p[dir]


    def get_pos_c_dist(self):
        for pos in self.pos_c:
            yield pos, self.pos_c[pos]


    def get_rel_c_dist(self):
        for rel in self.rel_c:
            yield rel, self.rel_c[rel]


    def get_dir_c_dist(self):
        for dir in self.dir_c:
            yield dir, self.dir_c[dir]


class Statistics(object):

    def __init__(self, corpus, gold_file, gold_order, sure_only=True, pos_voc_file=None, rel_voc_file=None):
        self.gold_aligned = defaultdict(lambda: defaultdict(set))
        self.stats = defaultdict(lambda: defaultdict(int))
        self.cond_freq = defaultdict(int)

        self.max_i = 0
        self.min_i = 0

        self.read_gold_file(gold_file, gold_order, sure_only=sure_only)
        self.cond_voc = CondIndex()
        self.read_corpus(corpus)
        self.array_length = abs(self.min_i) + self.max_i + 1
        self.make_arrays()
        self.pos_voc = dict()
        self.rel_voc = dict()
        self.dir_voc = {-1: "l", 1:  "r", None: "---"}

        if pos_voc_file:
            self.pos_voc = self.read_cond_voc_file(pos_voc_file)
        if rel_voc_file:
            self.rel_voc = self.read_cond_voc_file(rel_voc_file)



    def read_cond_voc_file(self, fname):
        voc = dict()
        voc[None] = "---"
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


        for sent_num, (e_toks, f_toks, f_heads, pos, rel, dir, order) in enumerate(corpus):
            # reorder to normal order
            order_indices, _  = zip(*sorted(list(enumerate(order)), key=lambda t: t[1]))
            f_toks = map(f_toks.__getitem__, order_indices)
            pos = map(pos.__getitem__, order_indices)
            rel = map(rel.__getitem__, order_indices)
            dir = map(dir.__getitem__, order_indices)
            f_heads = [order[f_heads[oi]] for oi in order_indices]

            J = len(f_toks)
            children = [set() for _ in xrange(J)]
            for j, h in enumerate(f_heads[1:]):
                children[h].add(j + 1)

            for j in xrange(J):
                parents_aligned = self.gold_aligned[sent_num][j]
                if not parents_aligned or (not children[j]):
                    continue
                p_par, r_par, d_par = pos[j], rel[j], dir[j]

                parent_weight = 1.0 / len(parents_aligned)
                for c in children[j]:
                    c_aligned = self.gold_aligned[sent_num][c]
                    p_c, r_c, d_c = pos[c], rel[c], dir[c]
                    for i_p in parents_aligned:
                        for i in c_aligned:
                            cond_id = self.cond_voc.add(p_par, r_par, d_par, p_c, r_c, d_c)
                            rel_dist = i - i_p
                            self.stats[cond_id][rel_dist] += parent_weight
                            self.cond_freq[cond_id] += parent_weight
                            if rel_dist > self.max_i:
                                self.max_i = rel_dist
                            if rel_dist < self.min_i:
                                self.min_i = rel_dist


    def make_arrays(self):
        for cond_id in self.stats:
            array = np.zeros(self.array_length)
            for i, j in enumerate(xrange(self.min_i, self.max_i+1)):
                array[i] = self.stats[cond_id][j]
            self.stats[cond_id] = array

    def get_dist(self, parent_con, tok_con):
        dist_dict = dict()
        freq_dict = dict()
        for comb, indices in self.cond_voc.get_dist(parent_con, tok_con).iteritems():
            count = 0
            array = np.zeros(self.array_length)
            for i in indices:
                array += self.stats[i]
                count += self.cond_freq[i]
            dist_dict[comb] = array
            freq_dict[comb] = count
        return dist_dict, freq_dict

    def compute_results(self, dist_dict):
        results = dict()
        for key, dist in dist_dict.iteritems():
            reorder_probs = np.array([np.sum(dist[:abs(self.min_i)]), dist[abs(self.min_i)], np.sum(dist[abs(self.min_i)+1:])])
            # smoothing:
            dist += (np.ones(len(dist)) * 0.0000001)
            dist = dist / np.sum(dist)
            reorder_probs += (np.ones(3) * 0.0000001)
            reorder_probs = reorder_probs / np.sum(reorder_probs)
            dist_logs = np.log(dist)
            reorder_logs = np.log(reorder_probs)
            dist_entropy = -np.sum(np.multiply(dist, dist_logs))
            reorder_entropy = -np.sum(np.multiply(reorder_probs, reorder_logs))
            results[key] = reorder_probs, reorder_entropy, dist_entropy
        return results

    def make_stats(self):
        combos = ["", "p", "r", "d", "pr", "pd", "rd", "prd"]
        aggregated_results = dict()
        for p_comb in combos:
            for c_comb in combos:
                if not p_comb and not c_comb:
                    continue
                print "Computing ", p_comb, c_comb
                dist, freq = self.get_dist(p_comb, c_comb)
                results = self.compute_results(dist)
                aggregated_reorder = 0
                aggregated_pos = 0
                norm = 0
                for key, v in results.iteritems():
                    f = freq[key]
                    norm += f
                    aggregated_reorder += f * v[1]
                    aggregated_pos += f * v[2]
                aggregated_pos = float(aggregated_pos) / norm
                aggregated_reorder = float(aggregated_reorder) / norm
                aggregated_results[p_comb, c_comb] = (aggregated_reorder, aggregated_pos)

        for k, v in sorted(aggregated_results.items(), key=lambda t: t[1][0]):
            print k, v
                # for (pp, pr, pd, cp, cr, cd), v in sorted(results.items(), key= lambda t: t[1][1]):
                #     f = freq[pp, pr, pd, cp, cr, cd]
                #     pp = self.pos_voc.get(pp, pp)
                #     pr = self.rel_voc.get(pr, pr)
                #     pd = self.dir_voc.get(pd, pd)
                #     cp = self.pos_voc.get(cp, cp)
                #     cr = self.rel_voc.get(cr, cr)
                #     cd = self.dir_voc.get(cd, cd)
                #
                #     print pp, pr, pd, cp, cr, cd, np.round(v[0], decimals=3), f



if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-gold", required=True)
    arg_parser.add_argument("-pos_voc", required=False, default="")
    arg_parser.add_argument("-rel_voc", required=False, default="")

    args = arg_parser.parse_args()

    stat = Statistics(args.corpus, args.gold, ("f", "e"), pos_voc_file=args.pos_voc, rel_voc_file=args.rel_voc)
    stat.make_stats()


# read gold file

# iterate through corpus

# for each parent
  # for each child
  # see if child is aligned, if yes, add statistics