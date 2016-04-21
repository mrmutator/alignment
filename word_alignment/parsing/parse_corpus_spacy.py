from word_alignment.utils.Corpus_Reader import GIZA_Reader
import codecs
import argparse

class POSVoc(object):

    def __init__(self):
        self.i2v = dict()
        self.v2i = dict()
        self.i = 1

    def get_id(self, w):
        if w not in self.v2i:
            i = self.i
            self.i2v[i] = w
            self.v2i[w] = i
            self.i += 1
        return self.v2i[w]

    def get_voc(self):
        string = ""
        for i in sorted(self.i2v):
            string += str(i) + "\t" + self.i2v[i] + "\n"
        return string


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-snt", required=True)
arg_parser.add_argument("-raw_f", required=True)
arg_parser.add_argument('-fix_punctuation', dest='fix_punctuation', action='store_true', default=False)
args = arg_parser.parse_args()

# at the moment only one parser supported
from Parser import Spacy_Parser
parser = Spacy_Parser(fix_punctuation=args.fix_punctuation)

outfile = codecs.open(args.snt + ".parsed", "w", "utf-8")
filter_file = open(args.snt + ".filter", "w")
pos_voc_file = open(args.snt + ".posvoc", "w")
rel_voc_file = open(args.snt + ".relvoc", "w")
corpus = GIZA_Reader(args.snt, alignment_order=('e', 'f'))
raw_infile = codecs.open(args.raw_f, "r", "utf-8")

pos_voc = POSVoc()
rel_voc = POSVoc()
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
        pos_tags = tree.source_pos_tags
        pos_tags = map(pos_tags.__getitem__, order)
        pos_tags = map(pos_voc.get_id, pos_tags)
        relations = tree.relations
        relations = map(relations.__getitem__, order)
        relations = map(rel_voc.get_id, relations)
        directions = tree.directions
        directions = map(directions.__getitem__, order)
        # test for weird cases (bug reported)
        test_order = sorted(order)
        if test_order != range(len(f_i)):
            skipped+= 1
            print "Skipped because of bug."
            filter_file.write(str(i)+"\n")
            continue
        _, f_heads = zip(*pairs)
        f = map(f_i.__getitem__, order)
        outfile.write(" ".join(map(str, e)) + "\n")
        outfile.write(" ".join(map(str, f)) + "\n")
        outfile.write(" ".join(map(str, f_heads)) + "\n")
        outfile.write(" ".join(map(str, pos_tags)) + "\n")
        outfile.write(" ".join(map(str, relations)) + "\n")
        outfile.write(" ".join(map(str, directions)) + "\n")
        outfile.write(" ".join(map(str, order)) + "\n\n")

    else:
        skipped += 1
        filter_file.write(str(i)+"\n")

outfile.close()
filter_file.close()
raw_infile.close()
pos_voc_file.write(pos_voc.get_voc())
pos_voc_file.close()
rel_voc_file.write(rel_voc.get_voc())
rel_voc_file.close()