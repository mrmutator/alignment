from CorpusReader import CorpusReader
import argparse


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-aligned_file", required=True)

    args = arg_parser.parse_args()
    corpus = CorpusReader(args.corpus)

    outfile = open(args.aligned_file + ".reordered", "w")
    with open(args.aligned_file, "r") as infile:
        for _, _, _, _, _, _, order in corpus:
            als = infile.readline().strip().split()
            if not als:
                break
            es, fs = zip(*[a.split("-") for a in als])
            fs = map(int, fs)
            reordered_fs = map(str, map(order.__getitem__, fs))
            als = zip(es, reordered_fs)
            outfile.write(" ".join(["-".join(a) for a in als]) + "\n")

    outfile.close()

