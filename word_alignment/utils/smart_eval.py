import argparse
from collections import defaultdict
import numpy as np

class CorpusReader(object):
    def __init__(self, corpus_file):
        self.corpus_file = open(corpus_file, "r")
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
                b = -1
                buffer = []


    def __iter__(self):
        return self.next()

    def get_length(self):
        c = 0
        for _ in self:
            c += 1
        return c


class Evaluation(object):
    def __init__(self, gold_file, gold_order=('e', 'f'), al_order=('e', 'f')):
        self.alignments = defaultdict(dict)
        self.sure = defaultdict(lambda: defaultdict(set))
        self.probable = defaultdict(lambda: defaultdict(set))
        self.gold_file = gold_file
        self.max_gold_sent = 0
        self.read_gold_file(gold_file, gold_order)


    def read_gold_file(self, file_name, order=("e", "f")):

        with open(file_name, "r") as infile:
            for line in infile:
                els = line.strip().split()
                snt = els[0]
                al_type = els[3]
                if order == ("e", "f"):
                    e = els[1]
                    f = els[2]
                elif order == ("f", "e"):
                    e = els[2]
                    f = els[1]
                else:
                    raise Exception("Invalid order type.")

                snt, e, f = int(snt), int(e), int(f)
                if al_type == "P":
                    self.probable[snt][f].add(e)
                if al_type == "S":
                    self.sure[snt][f].add(e)
                    self.probable[snt][f].add(e)
                self.max_gold_sent = snt

    def read_alignment_file(self, file_name, order=('e', 'f')):
        self.alignments = defaultdict(dict)
        self.alignment_file = file_name
        self.aligned_f = set()
        with open(file_name, "r") as infile:
            for i, line in enumerate(infile):
                if i + 1 > self.max_gold_sent:
                    break
                als = [tuple(x.split("-")) for x in line.strip().split()]
                for al in als:
                    if order == ("e", "f"):
                        tpl = (i + 1, int(al[0]) + 1, int(al[1]) + 1)
                    elif order == ("f", "e"):
                        tpl = (i + 1, int(al[1]) + 1, int(al[0]) + 1)
                    else:
                        raise Exception("Invalid order type.")
                    self.alignments[tpl[0]][tpl[2]] = tpl[1]



    def compute_smart_eval_root(self, psnt_file):
        psnt = CorpusReader(args.psnt)

        correct = 0
        p_total = 0
        r_total = 0

        for i, (e_toks, f_toks, f_heads, pos, rel, hmm_trans, order) in enumerate(psnt):
            snt = i + 1
            if snt > self.max_gold_sent:
                break

            root = order[0] + 1

            if root in self.alignments[snt]:
                e = self.alignments[snt][root]
            else:
                e = 0

            gold_es = self.probable[snt][root]


            # precision
            if e > 0:
                if e in gold_es:
                    correct += 1
                p_total += 1

            # recall
            if len(gold_es) > 0:
                r_total += 1


        precision = float(correct) / p_total
        recall = float(correct) / r_total
        f_measure = (2 * precision * recall) / (precision + recall)

        return  " ".join(map(str, [precision, recall, f_measure]))

    def compute_smart_eval_first(self, psnt_file):
        psnt = CorpusReader(args.psnt)

        correct = 0
        p_total = 0
        r_total = 0

        for i, (e_toks, f_toks, f_heads, pos, rel, hmm_trans, order) in enumerate(psnt):
            snt = i + 1
            if snt > self.max_gold_sent:
                break

            root = 1

            if root in self.alignments[snt]:
                e = self.alignments[snt][root]
            else:
                e = 0

            gold_es = self.probable[snt][root]


            # precision
            if e > 0:
                if e in gold_es:
                    correct += 1
                p_total += 1

            # recall
            if len(gold_es) > 0:
                r_total += 1


        precision = float(correct) / p_total
        recall = float(correct) / r_total
        f_measure = (2 * precision * recall) / (precision + recall)

        return  " ".join(map(str, [precision, recall, f_measure]))


    def compute_smart_eval_real_root(self, psnt_file, real_root_lbl):
        psnt = CorpusReader(args.psnt)

        correct = 0
        p_total = 0
        r_total = 0

        for i, (e_toks, f_toks, f_heads, pos, rel, hmm_trans, order) in enumerate(psnt):
            snt = i + 1
            if snt > self.max_gold_sent:
                break

            try:
                root_j = rel.index(real_root_lbl)
            except ValueError:
                root_j = 0
            root = order[root_j] + 1

            if root in self.alignments[snt]:
                e = self.alignments[snt][root]
            else:
                e = 0

            gold_es = self.probable[snt][root]

            # precision
            if e > 0:
                if e in gold_es:
                    correct += 1
                p_total += 1

            # recall
            if len(gold_es) > 0:
                r_total += 1

        precision = float(correct) / p_total
        recall = float(correct) / r_total
        f_measure = (2 * precision * recall) / (precision + recall)

        return " ".join(map(str, [precision, recall, f_measure]))

    def compute_nl(self, psnt_file):
        psnt = CorpusReader(args.psnt)

        correct = np.zeros(2, dtype=np.float)
        p_total = np.zeros(2, dtype=np.float)
        r_total = np.zeros(2, dtype=np.float)

        for i, (e_toks, f_toks, f_heads, pos, rel, hmm_trans, order) in enumerate(psnt):
            snt = i + 1
            if snt > self.max_gold_sent:
                break

            for jp in xrange(1, len(f_toks)):
                j = order[jp]
                l = 0
                if order[f_heads[jp]] == order[jp] -1:
                    l = 1

                if j in self.alignments[snt]:
                    e = self.alignments[snt][j]
                else:
                    e = 0

                gold_es = self.probable[snt][j]

                # precision
                if e > 0:
                    if e in gold_es:
                        correct[l] += 1
                    p_total[l] += 1

                # recall
                if len(gold_es) > 0:
                    r_total[l] += 1

        precision = correct / p_total
        recall = correct / r_total
        f_measure = (2 * precision * recall) / (precision + recall)

        return " ".join(map(str, precision)) + " " + " ".join(map(str, recall)) + " " + " ".join(map(str, f_measure))


    def compute_source_length(self, psnt_file):
        psnt = CorpusReader(args.psnt)

        correct = defaultdict(int)
        p_total = defaultdict(int)
        r_total = defaultdict(int)

        for i, (e_toks, f_toks, f_heads, pos, rel, hmm_trans, order) in enumerate(psnt):
            snt = i + 1
            if snt > self.max_gold_sent:
                break


            for jp in xrange(len(f_heads)):
                j = order[jp] + 1
                if jp == 0:
                    l = 0
                else:
                    l = order[jp] - order[f_heads[jp]]


                if j in self.alignments[snt]:
                    e = self.alignments[snt][j]
                else:
                    e = 0

                gold_es = self.probable[snt][j]

                # precision
                if e > 0:
                    if e in gold_es:
                        correct[l] += 1
                    p_total[l] += 1

                # recall
                if len(gold_es) > 0:
                    r_total[l] += 1


        thresholds = [0, 5, 10, 15, 20, np.inf]

        agg_correct = np.zeros(len(thresholds)-1, dtype=np.float)
        agg_total_p = np.zeros(len(thresholds)-1, dtype=np.float)
        agg_total_r = np.zeros(len(thresholds)-1, dtype=np.float)

        for i, t in enumerate(thresholds[:-1]):
            for l in set(p_total.keys() + r_total.keys()):
                if l > t: # and l <= thresholds[i+1]:
                    agg_correct[i] += correct[l]
                    agg_total_p[i] += p_total[l]
                    agg_total_r[i] += r_total[l]




        precision = agg_correct / agg_total_p
        recall = agg_correct / agg_total_r
        f_measure = (2 * precision * recall) / (precision + recall)

        return " ".join([" ".join(map(str, precision)), " ".join(map(str, recall)), " ".join(map(str, f_measure)), " ".join(map(str, agg_correct))])



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-gold", required=True)
    arg_parser.add_argument('-psnt', required=True)
    arg_parser.add_argument('-root', required=True, type=int)
    arg_parser.add_argument('-test_files', nargs='+')
    arg_parser.add_argument('-swap_order', dest='swap_order', action='store_true', default=False)

    args = arg_parser.parse_args()

    if args.swap_order:
        gold_order = ('f', 'e')
    else:
        gold_order = ('e', 'f')


    eval = Evaluation(args.gold, gold_order=gold_order)
    for fname in args.test_files:
        eval.read_alignment_file(fname)
        print fname, "first", eval.compute_smart_eval_first(args.psnt)
        print fname, "root", eval.compute_smart_eval_root(args.psnt)
        print fname, "real", eval.compute_smart_eval_real_root(args.psnt, args.root)
        print fname, "nl", eval.compute_nl(args.psnt)
        print fname, "sl", eval.compute_source_length(args.psnt)