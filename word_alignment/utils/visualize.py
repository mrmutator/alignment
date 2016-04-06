from word_alignment.utils.Corpus_Reader import Corpus_Reader
import codecs
import subprocess
import tempfile
import shutil
import os
import re
from collections import defaultdict
import math

HEADER = R"""\documentclass[class=minimal,border=0pt]{standalone}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{tikz}
\usepackage{color}
\usetikzlibrary{shapes.geometric, arrows, positioning}
\tikzstyle{word} = [text centered]
\tikzstyle{correct} = [thick, green]
\tikzstyle{wrong} = [thick, red]
\tikzstyle{possible} = [dotted]
\tikzstyle{sure} = [dashed]
\tikzstyle{link} = [thick]
\begin{document}
\tikzstyle{my below of} = [below=of #1.south]
\tikzstyle{my right of} = [right=of #1.east]
\tikzstyle{my left of} = [left=of #1.west]
\tikzstyle{my above of} = [above=of #1.north]
\begin{tikzpicture}[auto, node distance=.7cm,>=latex']"""

FOOTER = r"""\end{tikzpicture}
\end{document}"""

ESCAPE = r'(&|%|\$|#|_|{|}|~|\^)'

class GoldFile(object):

    def __init__(self, file_name, order=("e", "f")):
        self.sure = defaultdict(set)
        self.probable = defaultdict(set)
        self.read_gold_file(file_name, order)

    def read_gold_file(self, file_name, order=("e","f")):
        with open(file_name, "r") as infile:
            for line in infile:
                els = line.strip().split()
                snt = int(els[0])
                al_type = els[3]
                if order == ("e", "f"):
                    e = els[1]
                    f = els[2]
                elif order == ("f", "e"):
                    e = els[2]
                    f = els[1]
                else:
                    raise Exception("Invalid order type.")

                tpl = (int(e)-1, int(f)-1)
                if al_type == "P":
                    self.probable[snt].add(tpl)
                if al_type == "S":
                    self.sure[snt].add(tpl)
    def get_gold_alignments(self, snt):
        return self.sure[snt], self.probable[snt]

def analyze_alignments(als, sure, probable):
    correct = set()
    wrong = set()
    for al in als:
        al_correct = False
        if al in sure:
            al_correct = True
            correct.add(al)
            sure.remove(al)
        if al in probable:
            al_correct = True
            correct.add(al)
            probable.remove(al)
        if not al_correct:
            wrong.add(al)
    return correct, wrong, sure, probable



def escape(string):
    string = re.sub(r"\\", "\\textbackslash", string)
    string = re.sub(r"\^", "\\^{}", string)
    return re.sub(ESCAPE, r"\\\1", string)

def make_node(tok, pos, side):
    tok = escape(tok)
    return "\\node (%s%d) [word, my right of=%s%d] {%s};\n" % (side, pos, side, pos-1, tok)

def make_link(e_pos, f_pos, typ="link"):
    return "\\draw [%s] (f%d) -- (e%d);\n" % (typ, f_pos, e_pos)

def make_deps(heads):
    deg_unit = 70.0 / (len(heads)-1)
    string = ""
    for i, h in enumerate(heads):
        if h == -1:
            string += "\\draw (root) edge[->] node {} (f%s);\n" % i
            continue
        deg = round((abs(h-i) * deg_unit) * (1.0/math.sqrt(abs(h-i))) + 20, 4)
        if h > i:
            string += "\\draw (f%s) edge[->, bend right=%s] node {} (f%s);\n" % (h,deg, i)
        else:
            string += "\\draw (f%s) edge[->, bend left=%s] node {} (f%s);\n" % (h,deg, i)
    return string

def visualize(e,f,a, heads=False, gold=None):
    if heads:
        root = heads.index(-1)
    string = "\\node (f0) [word] {%s};\n" % f[0]
    string += "\\node (e0) [word, my below of=f0] {%s};\n" % e[0]
    for i, f_tok in enumerate(f[1:]):
        string += make_node(f_tok, i+1, "f")
    for i, e_tok in enumerate(e[1:]):
        string += make_node(e_tok, i+1, "e")
    if gold:
        sure, probable = gold
        correct, wrong, sure, probable = analyze_alignments(a, sure, probable)
        for coll, typ in [(correct, "correct"), (wrong, "wrong"), (sure, "sure"), (probable, "possible")]:
            for e_pos, f_pos in coll:
                string += make_link(e_pos, f_pos, typ=typ)
    else:
        for e_pos, f_pos in a:
            string += make_link(e_pos, f_pos)
    if heads:
        string += "\\node (root) [word, my above of=f%s] {ROOT};\n" % root
        string += make_deps(heads)


    return string

def make_image(code, file_name):
    target_name = file_name
    file_name = file_name.split("/")[-1]
    current = os.getcwd()
    temp = tempfile.mkdtemp()
    os.chdir(temp)

    outfile = codecs.open(file_name + ".tex", "w", "utf-8")
    outfile.write(HEADER + code + FOOTER)
    outfile.close()

    proc=subprocess.Popen(['pdflatex',file_name + ".tex"])
    proc.communicate()

    os.chdir(current)
    os.rename(temp + "/" + file_name + ".pdf",target_name)
    shutil.rmtree(temp)

def visualize_all(corpus, file_name, max_sent_length=0, has_heads=False, gold_alignments=None):
    current = os.getcwd()
    temp = tempfile.mkdtemp()
    os.chdir(temp)

    files = []
    for i, (a,e,f) in enumerate(corpus):
        if max_sent_length and max(len(e), len(f)) > max_sent_length:
            continue
        heads = None
        if has_heads:
            f, heads = zip(*[t.split("_") for t in f])
            heads = map(int, heads)
        else:
            if "_" in f[0]:
                raise Exception("F has head annotations. Use -dep option.")
        gold = []
        if gold_alignments:
            gold = gold_alignments.get_gold_alignments(i+1)
        code = visualize(e,f,a, heads=heads, gold=gold)
        make_image(code, "aligned_%d.pdf" % i)
        files.append("aligned_%d.pdf" % i)

    proc = subprocess.Popen(["pdftk"] + files + ["cat", "output", "merged.pdf"])
    proc.communicate()

    os.chdir(current)
    os.rename(temp + "/" + "merged.pdf", file_name)
    shutil.rmtree(temp)


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("e_file")
    arg_parser.add_argument("f_file")
    arg_parser.add_argument("a_file")
    arg_parser.add_argument("out")
    arg_parser.add_argument("-al_order", default="ef")
    arg_parser.add_argument("-limit", default=0, type=int)
    arg_parser.add_argument("-max_length", default=0, type=int)
    arg_parser.add_argument('-dep', dest='dep', action='store_true', default=False)
    arg_parser.add_argument('-gold', default="", required=False)
    arg_parser.add_argument('-gold_order', default="ef", required=False)


    args = arg_parser.parse_args()

    if args.al_order == "ef":
        alignment_order = ("e", "f")
    elif args.al_order == "fe":
        alignment_order = ("f", "e")
    else:
        raise Exception("Bad al_order argument. Either 'ef' or 'fe'.")

    if args.limit == 0:
        limit = None
    else:
        limit = args.limit

    corpus = Corpus_Reader(args.e_file, args.f_file, args.a_file, alignment_order=alignment_order, limit=limit, strings="unicode")

    if args.gold:
        if args.gold_order == "ef":
            gold_order = ("e", "f")
        elif args.gold_order == "fe":
            gold_order = ("f", "e")
        else:
            raise Exception("Bad gold order argument. Either 'ef' or 'fe'.")
        gold = GoldFile(args.gold, order=gold_order)
    else:
        gold = None

    visualize_all(corpus, args.out, max_sent_length=args.max_length, has_heads=args.dep, gold_alignments=gold)
