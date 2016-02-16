__author__ = 'rwechsler'


import argparse



arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("filter_file")
arg_parser.add_argument("gold_file")

args = arg_parser.parse_args()

filtered = set()
with open(args.filter_file, "r") as infile:
    for line in infile:
        filtered.add(int(line.strip()))

outfile = open(args.gold_file + ".filtered", "w")
infile = open(args.gold_file, "r")

c = 0
last_seen = -1
for line in infile:
    els = line.strip().split()
    if int(els[0]) in filtered:
        continue
    if int(els[0]) > last_seen:
        c += 1
        last_seen = int(els[0])
    outfile.write(str(c) + " " + " ".join(els[1:]) + "\n")


infile.close()
outfile.close()




