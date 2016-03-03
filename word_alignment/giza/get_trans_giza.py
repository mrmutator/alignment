import codecs
import argparse

def read_voc(voc_file):
    voc_dict = dict()
    with codecs.open(voc_file, "r", "utf-8") as infile:
        for line in infile:
            i, w, _ = line.strip().split()
            voc_dict[int(i)] = w
    return voc_dict

def transform_t_table(t_file, out_file_name, e_dict, f_dict):
    outfile = codecs.open(out_file_name, "w", "utf-8")
    with open(t_file, "r") as infile:
        for line in infile:
            e, f, p = line.strip().split()
            if int(e) == 0:
                e_t = "NULL"
            else:
                e_t = e_dict[int(e)]
            f_t = f_dict[int(f)]
            outfile.write(" ".join([e_t, f_t, p]) + "\n")
    outfile.close()

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-voc_e", required=True)
arg_parser.add_argument("-voc_f", required=True)
arg_parser.add_argument("-t_table", required=True)

args = arg_parser.parse_args()


e_dict = read_voc(args.voc_e)
f_dict = read_voc(args.voc_f)

transform_t_table(args.t_table, args.t_table + ".words", e_dict, f_dict)