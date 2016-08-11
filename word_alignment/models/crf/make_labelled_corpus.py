import argparse
from CorpusReader import CorpusReader
from collections import defaultdict

class LazyFile(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.f = None

    def write(self, buffer):
        self.f = open(self.file_name, "wb")
        self.f.write(buffer)
        self.write = self.__write

    def __write(self, buffer):
        self.f.write(buffer)

    def close(self):
        if self.f:
            self.f.close()

class GoldFile(object):

    def __init__(self, file_name, order="ef", sure_only=True):
        self.f = open(file_name, "r")
        self.next = self.__iter_sent()
        assert  order=="ef" or order=="fe"
        if order == "ef":
            self.order = self.order_ef
        else:
            self.order = self.order_fe
        self.sure_only = sure_only

    def order_ef(self, x, y):
        return int(x), int(y)

    def order_fe(self, x, y):
        return int(y), int(x)


    def reset(self):
        self.f.seek(0)

    def __iter_sent(self):
        self.reset()
        buffer = []
        prev = 1
        for line in self.f:
            snt, x, y, t = line.split()
            e, f = self.order(x,y)
            if int(snt) == prev:
                if t == "S" or not self.sure_only:
                    buffer.append((e,f))
            else:
                yield buffer
                buffer = []
                if t == "S" or not self.sure_only:
                    buffer.append((e,f))
                prev = int(snt)

        yield buffer

    def __iter__(self):
        return self.__iter_sent()


def split_data_get_parameters(corpus, gold_file, file_prefix, num_sentences):
    subset_id = 1
    outfile_corpus = LazyFile(file_prefix + "." + str(subset_id) + ".sub_feat")
    order_file = open(file_prefix + ".order", "w")
    subset_c = 0
    for gold_als in gold_file:
        e_toks, f_toks, f_heads, pos, rel, hmm_transitions, order = corpus.next()
        subset_c += 1

        als = {f:set() for f in xrange(1, len(f_toks)+1)}
        for (e,f) in gold_als:
            als[f].add(e)

        alignment = []
        for f in xrange(1, len(f_toks)+1):
            if not als[f]:
                alignment.append(0)
            else:
                alignment.append(min(als[f]))

        # produce subcorpus file
        outfile_corpus.write(" ".join(map(str, e_toks)) + "\n")
        outfile_corpus.write(" ".join(map(str, f_toks)) + "\n")
        outfile_corpus.write(" ".join(map(str, f_heads)) + "\n")
        outfile_corpus.write(" ".join(map(str, pos)) + "\n")
        outfile_corpus.write(" ".join(map(str, rel)) + "\n")
        outfile_corpus.write(" ".join(map(str, hmm_transitions)) + "\n")
        outfile_corpus.write(" ".join(map(str, order)) + "\n")
        outfile_corpus.write(" ".join(map(str, alignment)) + "\n")
        outfile_corpus.write("\n")
        order_file.write(" ".join(map(str, order)) + "\n")


        if subset_c == num_sentences:
            subset_id += 1
            subset_c = 0
            outfile_corpus.close()
            outfile_corpus = open(file_prefix + "." + str(subset_id) + ".sub_feat", "w")

    outfile_corpus.close()
    order_file.close()





if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-gold", required=True)
    arg_parser.add_argument("-output_prefix", required=True)
    arg_parser.add_argument("-gold_order", required=True, type=str, default="ef")
    arg_parser.add_argument("-group_size", required=False, type=int, default=-1)
    args = arg_parser.parse_args()

    corpus = CorpusReader(args.corpus)
    gold_file = GoldFile(args.gold, order=args.gold_order)

    split_data_get_parameters(corpus, gold_file, args.output_prefix, args.group_size)
