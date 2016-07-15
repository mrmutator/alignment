import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("alignment_file")

args = arg_parser.parse_args()

outfile = open(args.alignment_file + ".swapped", "w")
with open(args.alignment_file, "r") as infile:
    for line in infile:
        als = line.split()
        als = [al.split("-") for al in als]
        als = ["-".join((al[1], al[0])) for al in als]
        outfile.write(" ".join(als) + "\n")
outfile.close()


