import argparse
import re

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("alignment_file")

args = arg_parser.parse_args()

infile = open(args.alignment_file, "r")
outfile = open(args.alignment_file + ".aligned", "w")
for line in infile:
    if line.startswith("NULL "):
        alignment = []
        als = re.findall("\(\{(.*?)\}\)", line.strip())
        for i, entry in enumerate(als[1:]):
            entry = entry.strip()
            if entry:
                for j in entry.split(" "):
                    alignment.append((i, int(j)-1))

        alignment = [str(i)+"-"+str(j)for (i,j) in alignment]
        outfile.write(" ".join(alignment) + "\n")

infile.close()
outfile.close()
