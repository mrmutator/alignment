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


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)

    args = arg_parser.parse_args()

    corpus = CorpusReader(args.corpus)

    Is = set()
    rels = set()
    total_max_tl = 0
    ls = set()
    total_children = set()
    for e_toks, f_toks, f_heads, _, rel, _, order in corpus:
        I = len(e_toks)
        Is.add(I)
        for j in xrange(len(f_toks)):
            rels.add(rel[j])

        children = [0 for _ in xrange(len(f_toks))]
        for j, h in enumerate(f_heads):
            if j != 0:
                children[h] += 1
        for c in children:
            total_children.add(c)

        tree_levels = [0] * len(f_toks)
        for j in xrange(1, len(f_toks)):
            tree_levels[j] = tree_levels[f_heads[j]] + 1
            l = order[j] - order[f_heads[j]]
            ls.add(l)
        max_tl = max(tree_levels)
        if max_tl > total_max_tl:
            total_max_tl = max_tl


    for typ in ["ftest2", "mixedext", "start", "mixed","l"]:

        with open(typ + ".features", "w") as outfile:
            for i in sorted(Is):
                outfile.write("j 0\tI "+str(i) + "\n")
            if typ == "ftest":
                for tl in xrange(total_max_tl +1):
                    outfile.write("ctl " + str(tl) + "\n")
            if typ in ["ftest", "rel"]:
                for r in rels:
                    outfile.write("crel " + str(r) + "\n")
            if typ == "l":
                for l in ls:
                    outfile.write("l " + str(l) + "\n")
            if typ == "ftest2":
                for tl in xrange(total_max_tl +1):
                    outfile.write("ctl " + str(tl) + "\n")
                    outfile.write("ptl " + str(tl) + "\n")
                for l in ls:
                    outfile.write("l " + str(l) + "\n")
                for c in total_children:
                    outfile.write("plc " + str(c) + "\n")
                    outfile.write("prc " + str(c) + "\n")
                    outfile.write("pc " + str(c) + "\n")
                    outfile.write("clc " + str(c) + "\n")
                    outfile.write("crc " + str(c) + "\n")
                    outfile.write("cc " + str(c) + "\n")

            if typ == "mixedext":
                outfile.write("phmm 0\n")
                outfile.write("phmm 1\n")
                outfile.write("chmm 0\n")
                outfile.write("chmm 1\n")
                outfile.write("phmm 0\tchmm 0\n")
                outfile.write("phmm 0\tchmm 1\n")
                outfile.write("phmm 1\tchmm 0\n")
                outfile.write("phmm 1\tchmm 1\n")
                for tl in xrange(total_max_tl + 1):
                    outfile.write("ctl " + str(tl) + "\n")
                    outfile.write("ptl " + str(tl) + "\n")
                for l in ls:
                    outfile.write("l " + str(l) + "\n")
                for c in total_children:
                    outfile.write("plc " + str(c) + "\n")
                    outfile.write("prc " + str(c) + "\n")
                    outfile.write("pc " + str(c) + "\n")
                    outfile.write("clc " + str(c) + "\n")
                    outfile.write("crc " + str(c) + "\n")
                    outfile.write("cc " + str(c) + "\n")
            if typ == "mixed":
                outfile.write("phmm 0\n")
                outfile.write("phmm 1\n")
                outfile.write("chmm 0\n")
                outfile.write("chmm 1\n")
                outfile.write("phmm 0\tchmm 0\n")
                outfile.write("phmm 0\tchmm 1\n")
                outfile.write("phmm 1\tchmm 0\n")
                outfile.write("phmm 1\tchmm 1\n")



