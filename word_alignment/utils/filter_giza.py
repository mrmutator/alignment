# Filters a GIZA SNT file according to a list of sentence numbers that must be removed.
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("filter_file")
arg_parser.add_argument("giza_snt_file")

args = arg_parser.parse_args()

filtered = set()
with open(args.filter_file, "r") as infile:
    for line in infile:
        if line.strip():
            filtered.add(int(line.strip()))

outfile = open(args.giza_snt_file + ".filtered", "w")
infile = open(args.giza_snt_file, "r")


c = 1
buffer = [infile.readline(), infile.readline(), infile.readline()]

while buffer[0]:
    if c not in filtered:
        outfile.write("".join(buffer))
    buffer = [infile.readline(), infile.readline(), infile.readline()]
    c += 1

infile.close()
outfile.close()




