import argparse


def gold_reader(a_file):
    buffer = []
    curr_id = 1
    for line in a_file:
        sent_id, e, f, _ =line.split()
        if int(sent_id) > curr_id:
            yield buffer
            buffer = []
            curr_id += 1
        buffer.append((int(e)-1,int(f)-1))
    if buffer:
        yield buffer


def analyse_giza_format(a_file, e_file, f_file):
    a_file.seek(0)
    message = "unclear"
    while True:
        es, fs = zip(*[map(int, a.split("-")) for a in a_file.readline().split()])
        max_e, max_f = max(es), max(fs)
        if f_file:
            f_len = len(f_file.readline().split())
            if max_f >= f_len:
                message = "Order is F-E"
                break
            if max_e >= f_len:
                message = "Order is E-F"
                break
        if e_file:
            e_len = len(e_file.readline().split())
            if max_e >= e_len:
                message = "Order is F-E"
                break
            if max_f >= e_len:
                message = "Order is E-F"
                break
    return message


def analyse_gold_format(a_file, e_file, f_file):
    a_file.seek(0)
    a_reader = gold_reader(a_file)
    message = "unclear"
    for als in a_reader:
        es, fs = zip(*als)
        max_e, max_f = max(es), max(fs)
        if f_file:
            f_len = len(f_file.readline().split())
            if max_f >= f_len:
                message = "Order is F-E"
                break
            if max_e >= f_len:
                message = "Order is E-F"
                break
        if e_file:
            e_len = len(e_file.readline().split())
            if max_e >= e_len:
                message = "Order is F-E"
                break
            if max_f >= e_len:
                message = "Order is E-F"
                break
    return message



arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("alignments")
arg_parser.add_argument("-f", required=False, default="")
arg_parser.add_argument("-e", required=False, default="")

args = arg_parser.parse_args()

e_file = None
f_file = None

if args.e:
    e_file = open(args.e, "r")
else:
    f_file = open(args.f, "r")
if not (e_file or f_file):
    raise Exception("Provide either -f or -e file")

a_file = open(args.alignments, "r")

a_line = a_file.readline()
if "-" in a_line:
    print analyse_giza_format(a_file, e_file, f_file)
else:
    print analyse_gold_format(a_file, e_file, f_file)
