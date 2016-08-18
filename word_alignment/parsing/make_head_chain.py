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
    arg_parser.add_argument("-corpus_file", required=True)
    args = arg_parser.parse_args()
    corpus = CorpusReader(args.corpus_file)


    outfile = open(args.corpus_file + ".headchain", "w")

    for e_toks, f_toks, f_heads, pos, rel, _, order in corpus:
        f_heads = [0] + range(0, len(f_toks)-1)
        outfile.write(" ".join(map(str, e_toks)) + "\n")
        outfile.write(" ".join(map(str, f_toks)) + "\n")
        outfile.write(" ".join(map(str, f_heads)) + "\n")
        outfile.write(" ".join(map(str, pos)) + "\n")
        outfile.write(" ".join(map(str, rel)) + "\n")
        outfile.write(" ".join(map(str, [0]*len(f_toks))) + "\n")
        outfile.write(" ".join(map(str, order)) + "\n\n")

    outfile.close()