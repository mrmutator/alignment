import codecs
import subprocess
import tempfile
import shutil
import os
import re
from collections import defaultdict
import math

HEADER = R"""\nonstopmode
\documentclass[class=minimal,border=0pt]{standalone}
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

class DataReader(object):

    def __init__(self, e_file, f_file, alignment_files, labels=[], dep=[], alignment_order=('e', 'f'), limit=0):
        self.limit = limit
        self.e_file = codecs.open(e_file, "r", "utf-8")
        self.f_file = codecs.open(f_file, "r", "utf-8")
        self.alignment_files = [open(f, "r") for f in alignment_files]
        if len(labels) != len(self.alignment_files):
            self.labels = ["file " + str(i+1) for i in xrange(len(self.alignment_files))]
        else:
            self.labels = labels
        if len(dep) != len(self.alignment_files):
            self.dep = [True for _ in xrange(len(self.alignment_files))]
        else:
            self.dep = dep

        self.next = self.__iter_source_dep

        if alignment_order==('e', 'f'):
            self.order = self.__convert_to_int_e_f
        elif alignment_order==('f', 'e'):
            self.order = self.__convert_to_int_f_e
        else:
            raise Exception("No valid alignment order.")

    def reset(self):
        for al in self.alignment_files:
            al.seek(0)
        self.e_file.seek(0)
        self.f_file.seek(0)

    def get_f(self):
        f_split = self.f_file.readline().strip().split()
        try:
            toks, heads = zip(*[t.split("_") for t in f_split])
        except ValueError:
            toks = f_split
            heads = []
        return toks, map(int, heads)

    def get_als(self):
        return [map(self.order, re.findall("(\d+)-(\d+)", a.readline().strip())) for a in self.alignment_files]


    def __iter_source_dep(self):
        self.reset()

        e_line = self.e_file.readline()


        c = 0
        while(e_line):
            c += 1
            e_toks = e_line.strip().split()
            f_toks, heads = self.get_f()
            als = self.get_als()
            if self.limit and c > self.limit:
                break
            yield e_toks, f_toks, heads, als
            e_line = self.e_file.readline()


    def __iter__(self):
        return self.next()

    def __convert_to_int_e_f(self, tpl):
        return int(tpl[0]), int(tpl[1])

    def __convert_to_int_f_e(self, tpl):
        return int(tpl[1]), int(tpl[0])



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
    sure = set(sure)
    probable = set(probable)
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

def visualize(e,f,a, heads=[], gold=None):
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

def make_image(code, file_name, label=""):
    target_name = file_name
    file_name = file_name.split("/")[-1]
    current = os.getcwd()
    temp = tempfile.mkdtemp()
    os.chdir(temp)
    code += "\\node (label) [word, below of=e0] {\\fontsize{6}{15.0}\\selectfont %s};\n" % label
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

    labels = corpus.labels
    dep = corpus.dep

    all_files = []
    for i, (e, f, f_heads, als) in enumerate(corpus):
        files = []
        if max_sent_length and max(len(e), len(f)) > max_sent_length:
            continue
        gold = []
        if gold_alignments:
            gold = gold_alignments.get_gold_alignments(i+1)
        for a_i, a in enumerate(als):
            tmp_heads = []
            if dep[a_i]:
                tmp_heads = f_heads
            code = visualize(e,f,a, heads=tmp_heads, gold=gold)
            ftemp_name = "aligned_%d.%d.pdf" % (i, a_i)
            make_image(code, ftemp_name, label="Sent. " + str(i+1) + ", %s" % labels[a_i])
            files.append(ftemp_name)

        if len(files) > 1:
            merged_file_name = "merged.%d.pdf" % i
            # pdfjam aligned_0.pdf aligned_1.pdf --nup 1x2 --outfile test.pdf
            proc = subprocess.Popen(["pdfjam"] + files + ["--nup", "1x2", "--fitpaper",
                                     "true", "--outfile", merged_file_name])
            proc.communicate()
            all_files.append(merged_file_name)
        else:
            all_files.append(ftemp_name)

    proc = subprocess.Popen(["pdftk"] + all_files + ["cat", "output", "merged.pdf"])
    proc.communicate()

    os.chdir(current)
    os.rename(temp + "/" + "merged.pdf", file_name)
    shutil.rmtree(temp)


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("out")
    arg_parser.add_argument("e_file")
    arg_parser.add_argument("f_file")
    arg_parser.add_argument("a_files", nargs="+")
    arg_parser.add_argument("-al_order", default="ef")
    arg_parser.add_argument("-limit", default=0, type=int)
    arg_parser.add_argument("-max_length", default=0, type=int)
    arg_parser.add_argument('-gold', default="", required=False)
    arg_parser.add_argument('-gold_order', default="ef", required=False)
    arg_parser.add_argument('-labels', default="", required=False, nargs="+")
    arg_parser.add_argument('-dep', default="", required=False, nargs="+")


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

    labels = []
    if args.labels:
        labels = args.labels
    dep = []
    if args.dep:
        dep = map(bool, map(int, args.dep))


    corpus = DataReader(args.e_file, args.f_file, args.a_files, labels=labels, dep=dep, alignment_order=alignment_order, limit=args.limit)

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

    visualize_all(corpus, args.out, max_sent_length=args.max_length, gold_alignments=gold)