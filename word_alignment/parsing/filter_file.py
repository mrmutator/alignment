import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-filter_file", required=True)
arg_parser.add_argument("-file", required=True)

args = arg_parser.parse_args()


filter_list = set()
with open(args.filter_file, "r") as infile:
    for line in infile:
        filter_list.add(int(line.strip()))

outfile = open(args.file + ".filtered", "w")
with open(args.file, "r") as infile:
    for i, line in enumerate(infile):
        if i+1 in filter_list:
            continue
        else:
            outfile.write(line)

outfile.close()