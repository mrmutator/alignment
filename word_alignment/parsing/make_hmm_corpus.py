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

def reorder(data, order):
    """
    Order is a list with same length as data that specifies for each position of data, which rank it has in the new order.
    :param data:
    :param order:
    :return:
    """
    new_data = [None] * len(data)
    for i, j in enumerate(order):
        new_data[j] = data[i]
    return new_data

def hmm_reorder(f_toks, pos, rel, order):
    # HMM reorder
    J = len(f_toks)
    new_f_toks = reorder(f_toks, order)
    new_pos = reorder(pos, order)
    new_rel = reorder(rel, order)
    new_f_heads = [0] + range(J - 1)
    new_order = range(J)
    return new_f_toks, new_f_heads, new_pos, new_rel, new_order


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus_file", required=True)
    args = arg_parser.parse_args()
    corpus = CorpusReader(args.corpus_file)


    outfile = open(args.corpus_file + ".hmm", "w")

    for e_toks, f_toks, f_heads, pos, rel, _, order in corpus:
        f_toks, f_heads, pos, rel, order = hmm_reorder(f_toks, pos, rel, order)

        outfile.write(" ".join(map(str, e_toks)) + "\n")
        outfile.write(" ".join(map(str, f_toks)) + "\n")
        outfile.write(" ".join(map(str, f_heads)) + "\n")
        outfile.write(" ".join(map(str, pos)) + "\n")
        outfile.write(" ".join(map(str, rel)) + "\n")
        outfile.write(" ".join(map(str, [0]*len(f_toks))) + "\n")
        outfile.write(" ".join(map(str, order)) + "\n\n")

    outfile.close()