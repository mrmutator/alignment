import argparse

class Evaluation(object):

    def __init__(self, gold_file, alignment_file, gold_order=('e', 'f'), al_order=('e', 'f')):
        self.alignments = set()
        self.sure = set()
        self.probable = set()

        self.read_alignment_file(alignment_file, order=al_order)
        self.read_gold_file(gold_file, order=gold_order)


    def read_gold_file(self, file_name, order=("e","f")):
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


    def read_alignment_file(self, file_name, order=('e', 'f')):
        with open(file_name, "r") as infile:
            for i, line in enumerate(infile):
                als = [tuple(x.split("-")) for x in line.strip().split()]
                for al in als:
                    if order == ("e", "f"):
                        tpl = (i+1, int(al[0])+1, int(al[1])+1)
                    elif order == ("f", "e"):
                        tpl = (i+1, int(al[1])+1, int(al[0])+1)
                    else:
                        raise Exception("Invalid order type.")
                    self.alignments.add(tpl)

    def precision(self):
        return len(self.alignments.intersection(self.probable)) / float(len(self.alignments))

    def recall(self):
        return len(self.alignments.intersection(self.sure)) / float(len(self.sure))

    def f1_measure(self):
        p = self.precision()
        r = self.recall()
        return (2*p*r) / (p + r)

    def aer(self):
        return 1 - ((len(self.alignments.intersection(self.sure)) + len(self.alignments.intersection(self.probable)))
                    / float(len(self.alignments) + len(self.sure)))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-gold", required=True)
    arg_parser.add_argument('-test', required=True)
    arg_parser.add_argument('-swap_order', dest='swap_order', action='store_true', default=False)

    args = arg_parser.parse_args()

    if args.swap_order:
        gold_order = ('f', 'e')
    else:
        gold_order= ('e', 'f')

    eval = Evaluation(args.gold, args.test, gold_order=gold_order)
    print "GOLD:", args.gold
    print "TEST:", args.test
    print
    print "P:   ", round(eval.precision(), 4)
    print "R:   ", round(eval.recall(), 4)
    print "F1:  ", round(eval.f1_measure(), 4)
    print "AER: ", round(eval.aer(), 4)





