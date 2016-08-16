import argparse


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-order_file", required=True)
    arg_parser.add_argument("-aligned_file", required=True)

    args = arg_parser.parse_args()

    order_file = open(args.order_file, "r")
    outfile = open(args.aligned_file + ".reordered", "w")
    with open(args.aligned_file, "r") as infile:
        for line in infile:
            if not line.strip():
                outfile.write("\n")
                order_file.readline()
                continue
            order = map(int, order_file.readline().strip().split())
            als = line.strip().split()
            es, fs = zip(*[a.split("-") for a in als])
            fs = map(int, fs)
            reordered_fs = map(str, map(order.__getitem__, fs))
            als = zip(es, reordered_fs)
            outfile.write(" ".join(["-".join(a) for a in als]) + "\n")

    outfile.close()
    order_file.close()

