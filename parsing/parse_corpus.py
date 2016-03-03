import codecs
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("corpus")
arg_parser.add_argument("-parallel_file", required=False, default="")

args = arg_parser.parse_args()

# at the moment only one parser supported
from Parser import Spacy_Parser
parser = Spacy_Parser()

infile = codecs.open(args.corpus, "r", "utf-8")
order_file = open(args.corpus + ".order", "w")
outfile = codecs.open(args.corpus + ".parsed", "w", "utf-8")

if args.parallel_file:
    infile2 = open(args.parallel_file, "r")
    outfile2 = open(args.parallel_file + ".processed", "w")

for line in infile:
    assert "_" not in line
    tokens = line.strip().split()
    tree = parser.dep_parse(tokens)
    if tree:
        order, pairs = tree.traverse_with_heads()
        order_file.write(" ".join(map(str, order)) + "\n")
        outfile.write(" ".join([p[0] + "_" + str(p[1]) for p in pairs]) + "\n")
        if args.parallel_file:
            outfile2.write(infile2.readline())

    else:
        order_file.write("SKIPPED\n")
        if args.parallel_file:
            infile2.readline()

infile.close()
outfile.close()
order_file.close()