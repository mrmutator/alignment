import codecs

def read_pos_voc(f):
    pos_dict = dict()
    with open(f, "r") as infile:
        for line in infile:
            id, pos = line.strip().split()
            pos = pos.replace("$", "/d")
            pos_dict[int(id)] = pos
    return pos_dict


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
    arg_parser.add_argument("-pos_voc", required=False, default="")

    args = arg_parser.parse_args()

    if args.pos_voc:
        pos_dict = read_pos_voc(args.pos_voc)

    infile = codecs.open(args.f_raw, "r", "utf-8")
    outfile = codecs.open(args.f_raw + ".annotated", "w", "utf-8")
    corpus = CorpusReader(args.psnt_file)
    for _, _, heads, pos, _, _, order in corpus:
        toks = infile.readline().strip().split()
        assert len(toks) == len(heads)
        pairs = zip(heads, order)
        if args.pos_voc:
            pos_pairs = zip(pos, order)
            new_pos, _ = zip(*sorted(pos_pairs, key=lambda t: t[1]))
            new_pos = ["/" + pos_dict.get(t) for t in new_pos]
        else:
            new_pos = ["" for t in toks]
        new_heads, _ = zip(*sorted(pairs, key=lambda t: t[1]))
        new_heads = map(order.__getitem__, new_heads)
        new_heads[order[0]] = -1  # root
        annotated = [t + new_pos[i] +  "_" + str(new_heads[i]) for i, t in enumerate(toks)]
        outfile.write(" ".join(annotated) + "\n")

    infile.close()
    outfile.close()
