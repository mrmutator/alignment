import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("alignments")
arg_parser.add_argument("-swap")

args = arg_parser.parse_args()



outfile = open(args.alignments + ".gold", "w")
with open(args.alignments, "r") as infile:
    for i, line in enumerate(infile):
        als = line.split()
        for al in als:
            e, f = al.split("-")
            if args.swap:
                e, f = f, e
            outfile.write(" ".join([str(i+1), e, f, "S"])+"\n")

outfile.close()