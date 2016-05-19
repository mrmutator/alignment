import codecs
import re

class SubcorpusReader(object):
    def __init__(self, corpus_file, limit=None, return_order=False):
        self.corpus_file = open(corpus_file, "r")
        self.limit = limit
        self.next = self.__iter_sent
        if return_order:
            self.buffer_end = 5
        else:
            self.buffer_end = 4

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
            if b == 5:
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

def read_condvoc(f):
    hmt_params_dict = dict()
    with open(f, "r") as infile:
        for line in infile:
            id, data = line.strip().split("\t")
            id = int(id)
            m = re.search(".*, (False|True)\)", data)
            hmt_param = "1" if m.group(1) == "True" else "0"
            hmt_params_dict[id] = hmt_param
    return hmt_params_dict


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("f_raw")
    arg_parser.add_argument("subcorpus_file")
    arg_parser.add_argument("condvoc_file")

    args = arg_parser.parse_args()

    hmt_params_dict = read_condvoc(args.condvoc_file)
    hmt_params_dict[-1] = "0" # root is not hmm-parameter

    infile = codecs.open(args.f_raw, "r", "utf-8")
    outfile = codecs.open(args.f_raw + ".annotated", "w", "utf-8")
    outfile2 = open(args.f_raw + ".paramtypes", "w")
    corpus = SubcorpusReader(args.subcorpus_file, return_order=True)
    for _, _, heads, cons, order in corpus:
        toks = infile.readline().strip().split()
        assert len(toks) == len(heads)
        reordered_cons = [None] * len(toks)
        for i, j in enumerate(order):
            reordered_cons[j] = cons[i]
        param_annotation = map(hmt_params_dict.get, reordered_cons)
        pairs = zip(heads, order)
        new_heads, _ = zip(*sorted(pairs, key=lambda t: t[1]))
        new_heads = map(order.__getitem__, new_heads)
        new_heads[order[0]] = -1  # root
        annotated = [t + "_" + param_annotation[i] +  "_" + str(new_heads[i]) for i, t in enumerate(toks)]
        outfile.write(" ".join(annotated) + "\n")
        outfile2.write(" ".join(map(str, [param_annotation[i] for i, _ in enumerate(toks)])) + "\n")

    infile.close()
    outfile.close()
    outfile2.close()
