import codecs


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
    arg_parser.add_argument("f_raw")
    arg_parser.add_argument("psnt_file")

    args = arg_parser.parse_args()

    infile = codecs.open(args.f_raw, "r", "utf-8")
    outfile = codecs.open(args.f_raw + ".annotated", "w", "utf-8")
    corpus = CorpusReader(args.psnt_file)
    for _, _, heads, _, _, _, order in corpus:
        toks = infile.readline().strip().split()
        assert len(toks) == len(heads)
        pairs = zip(heads, order)
        new_heads, _ = zip(*sorted(pairs, key=lambda t: t[1]))
        new_heads = map(order.__getitem__, new_heads)
        new_heads[order[0]] = -1  # root
        annotated = [t + "_" + str(new_heads[i]) for i, t in enumerate(toks)]
        outfile.write(" ".join(annotated) + "\n")

    infile.close()
    outfile.close()
