from word_alignment.utils.Corpus_Reader import GIZA_Reader
import codecs
import argparse

class ParsedFile(object):

    def __init__(self, file_name):
        self.parsed_file = codecs.open(args.parsed_f, "r", "utf-8")

    def get_next_parse(self):
        string = self.parsed_file.readline()
        string += self.parsed_file.readline()
        for line in self.parsed_file:
            if line.strip():
                string += line
            else:
                break
        return string


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-snt", required=True)
arg_parser.add_argument("-parsed_f", required=True)
arg_parser.add_argument('-strict', dest='strict', action='store_true', default=False)

args = arg_parser.parse_args()

from Parser import StanfordParser
parser = StanfordParser()

outfile = codecs.open(args.snt + ".parsed", "w", "utf-8")
filter_file = open(args.snt + ".filtered", "w")
corpus = GIZA_Reader(args.snt, alignment_order=('e', 'f'))
parsed_file = ParsedFile(args.parsed_f)
skipped = 0
i = 0
for e, f_i in corpus:
    i += 1
    parse = parsed_file.get_next_parse()
    tree = parser.dep_parse(parse, strict=args.strict)
    if tree:
        order, pairs = tree.traverse_with_heads()
        # test for weird cases (bug reported)
        test_order = sorted(order)
        assert len(pairs) == len(f_i)
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
        outfile.write(" ".join(map(str, order)) + "\n\n")

    else:
        skipped += 1
        filter_file.write(str(i)+"\n")

outfile.close()
filter_file.close()