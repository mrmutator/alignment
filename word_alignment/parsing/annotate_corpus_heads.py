# annotate original F-corpus file with heads from snt-parsed (5-lines) file
import codecs

class CorpusReader(object):
    def __init__(self, corpus_file, limit=None, return_order=False):
        self.corpus_file = open(corpus_file, "r")
        self.limit = limit
        self.next = self.__iter_sent
        if return_order:
            self.buffer_end = 4
        else:
            self.buffer_end = 3

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
            if b == 4:
                yield buffer[:self.buffer_end]
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
    arg_parser.add_argument("snt_file")

    args = arg_parser.parse_args()

    infile = codecs.open(args.f_raw, "r", "utf-8")
    outfile = codecs.open(args.f_raw + ".annotated", "w", "utf-8")
    corpus = CorpusReader(args.snt_file, return_order=True)
    for _, _, heads, order in corpus:
        toks = infile.readline().strip().split()
        assert len(toks) == len(heads)
        pairs = zip(heads, order)
        new_heads, _ = zip(*sorted(pairs, key=lambda t: t[1]))
        new_heads = map(order.__getitem__, new_heads)
        new_heads[order[0]] = -1 # root
        annotated = [t + "_" + str(new_heads[i]) for i, t in enumerate(toks)]
        outfile.write(" ".join(annotated) + "\n")

    infile.close()
    outfile.close()


