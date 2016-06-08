from collections import defaultdict
import numpy as np

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

class ParentIndex(object):

    def __init__(self):
        self.i = 0
        self.data = dict()
        self.pos = defaultdict(set)
        self.rel = defaultdict(set)
        self.dir = defaultdict(set)


    def add(self, pos, rel, dir):
            if (pos, rel, dir) in self.data:
                return self.data[(pos, rel, dir)]
            else:
                self.i += 1
                self.data[(pos, rel, dir)] = self.i
                self.pos[pos].add(self.i)
                self.rel[rel].add(self.i)
                self.dir[dir].add(self.i)
                return self.data[(pos, rel, dir)]

    def get_pos_dist(self):
        for pos in self.pos:
            yield pos, self.pos[pos]

    def get_rel_dist(self):
        for rel in self.rel:
            yield rel, self.rel[rel]

    def get_dir_dist(self):
        for dir in self.dir:
            yield dir, self.dir[dir]


class Statistics(object):

    def __init__(self, corpus, gold_file, gold_order, sure_only=True):
        self.gold_aligned = defaultdict(lambda: defaultdict(set))
        self.stats = defaultdict(lambda: defaultdict(int))

        self.max_i = 0
        self.min_i = 0

        self.read_gold_file(gold_file, gold_order, sure_only=sure_only)
        self.cond_voc = ParentIndex()
        self.read_corpus(corpus)
        self.array_length = abs(self.min_i) + self.max_i + 1
        self.make_arrays()



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
            J = len(f_toks)
            children = [set() for _ in xrange(J)]
            for j, h in enumerate(f_heads[1:]):
                children[h].add(j + 1)

            for j in xrange(J):
                p, r, d = pos[j], rel[j], dir[j]
                cond_id = self.cond_voc.add(p, r, d)

                parents_aligned = self.gold_aligned[sent_num][j]
                if not parents_aligned:
                    continue
                parent_weight = 1.0 / len(parents_aligned)
                for c in children[j]:
                    c_aligned = self.gold_aligned[sent_num][c]
                    for i_p in parents_aligned:
                        for i in c_aligned:
                            rel_dist = i - i_p
                            self.stats[cond_id][rel_dist] += parent_weight
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
            print array

    def get_pos_dist(self):
        dist_dict = dict()
        for pos, indices in self.cond_voc.get_pos_dist():
            array = np.zeros(self.array_length)
            for i in indices:
                array += self.stats[i]
            dist_dict[pos] = array
        return dist_dict

    def get_rel_dist(self):
        dist_dict = dict()
        for rel, indices in self.cond_voc.get_rel_dist():
            array = np.zeros(self.array_length)
            for i in indices:
                array += self.stats[i]
            dist_dict[rel] = array
        return dist_dict

    def get_dir_dist(self):
        dist_dict = dict()
        for dir, indices in self.cond_voc.get_dir_dist():
            array = np.zeros(self.array_length)
            for i in indices:
                array += self.stats[i]
            dist_dict[dir] = array / np.sum(array)
        return dist_dict

    def compute_results(self, dist_dict):
        for key, dist in dist_dict.iteritems():
            reorder_probs = np.array([np.sum(dist[:abs(self.min_i)]), dist[abs(self.min_i)], np.sum(dist[abs(self.min_i)+1:])])
            dist_logs = np.log(dist)
            reorder_logs = np.log(reorder_probs)
            dist_entropy = -np.sum(np.multiply(dist, dist_logs))



if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-gold", required=True)

    args = arg_parser.parse_args()

    stat = Statistics(args.corpus, args.gold, ("f", "e"))


# read gold file

# iterate through corpus

# for each parent
  # for each child
  # see if child is aligned, if yes, add statistics