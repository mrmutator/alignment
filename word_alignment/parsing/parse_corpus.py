from word_alignment.utils.Corpus_Reader import GIZA_Reader
import codecs
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-snt", required=True)
arg_parser.add_argument("-raw_f", required=True)

args = arg_parser.parse_args()

# at the moment only one parser supported
from Parser import Spacy_Parser
parser = Spacy_Parser()

outfile = codecs.open(args.snt + ".parsed", "w", "utf-8")

corpus = GIZA_Reader(args.snt, alignment_order=('e', 'f'))
raw_infile = codecs.open(args.raw_f, "r", "utf-8")
skipped = 0
i = 0
for e, f_i in corpus:
    i += 1
    f_raw = raw_infile.readline()
    assert "_" not in f_raw
    f_raw = f_raw.strip().split()
    assert len(f_raw) == len(f_i)
    tree = parser.dep_parse(f_raw)
    if tree:
        order, pairs = tree.traverse_with_heads()
        _, f_heads = zip(*pairs)
        f = map(f_i.__getitem__, order)
        outfile.write(" ".join(map(str, e)) + "\n")
        outfile.write(" ".join(map(str, f)) + "\n")
        outfile.write(" ".join(map(str, f_heads)) + "\n")
        outfile.write(" ".join(map(str, order)) + "\n\n")

    else:
        skipped += 1
        print "skipped: ", i

outfile.close()
raw_infile.close()