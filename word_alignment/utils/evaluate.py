import argparse


class Evaluation(object):
    def __init__(self, gold_file, alignment_file=None, gold_order=('e', 'f'), al_order=('e', 'f'), paramtypes=None):
        self.alignments = set()
        self.sure = set()
        self.probable = set()
        self.gold_file = gold_file
        self.paramtypes = None

        if alignment_file:
            self.read_alignment_file(alignment_file, order=al_order)
            self.alignment_file = alignment_file
        self.read_gold_file(gold_file, order=gold_order)


        if paramtypes:
            self.read_paramtypes(paramtypes)

    def read_gold_file(self, file_name, order=("e", "f")):
        self.gold_aligned = set()
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

                tpl = (int(snt), int(e), int(f))
                if al_type == "P":
                    self.probable.add(tpl)
                if al_type == "S":
                    self.sure.add(tpl)
                    self.probable.add(tpl)
                self.gold_aligned.add((tpl[0], tpl[2]))

    def read_alignment_file(self, file_name, order=('e', 'f')):
        self.alignments = set()
        self.alignment_file = file_name
        self.aligned_f = set()
        with open(file_name, "r") as infile:
            for i, line in enumerate(infile):
                als = [tuple(x.split("-")) for x in line.strip().split()]
                for al in als:
                    if order == ("e", "f"):
                        tpl = (i + 1, int(al[0]) + 1, int(al[1]) + 1)
                    elif order == ("f", "e"):
                        tpl = (i + 1, int(al[1]) + 1, int(al[0]) + 1)
                    else:
                        raise Exception("Invalid order type.")
                    self.alignments.add(tpl)
                    self.aligned_f.add((i+1, tpl[2]))

    def read_paramtypes(self, fname):
        data = dict()
        self.all_types = set()
        with open(fname, "r") as infile:
            for i, line in enumerate(infile):
                data[i+1] = dict()
                ptypes = map(int, line.strip().split())
                for j, ptype in enumerate(ptypes):
                    data[i+1][j+1] = ptype
                    self.all_types.add(ptype)


        self.paramtypes = data

    def precision(self):
        return len(self.alignments.intersection(self.probable)) / float(len(self.alignments))

    def recall(self):
        return len(self.alignments.intersection(self.sure)) / float(len(self.sure))

    def f1_measure(self):
        p = self.precision()
        r = self.recall()
        return (2 * p * r) / (p + r)

    def aer(self):
        return 1 - ((len(self.alignments.intersection(self.sure)) + len(self.alignments.intersection(self.probable)))
                    / float(len(self.alignments) + len(self.sure)))

    def ptype_acc(self):
        null = set()
        null_gold = set()
        for sent in self.paramtypes:
            for f in self.paramtypes[sent]:
                if not ((sent, f) in self.aligned_f):
                    null.add((sent, 0, f))
                if not ((sent, f)) in self.gold_aligned:
                    null_gold.add((sent, 0, f))
        self.param_alignments = {}
        for k in self.all_types:
            self.param_alignments[k] = set()
        for (sent, e, f) in self.alignments.union(null):
            ptype = self.paramtypes[sent][f]
            self.param_alignments[ptype].add((sent, e, f))

        results = dict()
        for k,v in self.param_alignments.iteritems():
            results[k] = len(v.intersection(self.probable.union(null_gold))) / float(len(v))
        return results

    def print_machine_output(self):
        ptype_res = []
        if self.paramtypes:
            _, ptype_res = zip(*sorted(self.ptype_acc().items(), key=lambda t: t[0]))
        print "\t".join([self.alignment_file] + map("{:6.4f}".format, [round(self.precision(), 4),
                                                                       round(self.recall(), 4),
                                                                       round(self.f1_measure(), 4),
                                                                       round(self.aer(), 4)] + list(ptype_res)))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-gold", required=True)
    arg_parser.add_argument('-test_files', nargs='+')
    arg_parser.add_argument('-paramtypes', required=False, default="")
    arg_parser.add_argument('-swap_order', dest='swap_order', action='store_true', default=False)

    args = arg_parser.parse_args()

    if args.swap_order:
        gold_order = ('f', 'e')
    else:
        gold_order = ('e', 'f')

    eval = Evaluation(args.gold, gold_order=gold_order, paramtypes=args.paramtypes)
    for fname in args.test_files:
        eval.read_alignment_file(fname)
        eval.print_machine_output()
