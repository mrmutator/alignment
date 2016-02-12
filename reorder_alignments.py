import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("order_file")
arg_parser.add_argument("alignment_file")
arg_parser.add_argument('-swap_order', dest='swap_order', action='store_true', default=False)

args = arg_parser.parse_args()

order_file = open(args.order_file, "r")
infile = open(args.alignment_file, "r")
outfile = open(args.alignment_file + ".reordered", "w")

for line in infile:
    line_order = order_file.readline().strip()
    order = map(int, line_order.split())

    als = line.strip().split()
    als = [tuple(x.split("-")) for x in als]
    if args.swap_order:
        als = [(y,x) for x,y in als]

    reordered = []
    for e,f in als:
        new_e = order[int(e)]
        reordered.append((new_e,f))

    reordered = ["-".join([str(x), str(y)]) for x,y in reordered]
    outfile.write(" ".join(reordered) + "\n")

order_file.close()
infile.close()
outfile.close()

