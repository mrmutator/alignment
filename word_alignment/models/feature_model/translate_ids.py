import gzip
import argparse


def read_trans_file(fname):
    trans = dict()
    with open(fname, "r") as infile:
        for line in infile:
            original, new = map(int, line.split())
            trans[original] = new
    return trans


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("corpus_file")
    arg_parser.add_argument("-vec_file", required=True)
    arg_parser.add_argument("-con_file", required=True)

    args = arg_parser.parse_args()


    con_trans = read_trans_file(args.con_file)
    vec_trans = read_trans_file(args.vec_file)

    outfile = gzip.open(args.corpus_file + ".corrected.gz", "wb")
    with gzip.open(args.corpus_file, "rb") as infile:
        c = 0
        for line in infile:
            c += 1
            if line.strip() == "":
                c = 0
            if c > 3:
                els = map(int, line.split())
                con_id = con_trans[els[0]]
                vec_ids = map(vec_trans.__getitem__, els[1:])
                line = " ".join(map(str, [con_id] + vec_ids)) + "\n"
            outfile.write(line)
    outfile.close()