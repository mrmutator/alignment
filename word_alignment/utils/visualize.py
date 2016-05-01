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
\tikzstyle{wordnull} = [text centered, blue]
\tikzstyle{wordnullwrong} = [text centered, red]
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

    def __init__(self, e_file, f_files, alignment_files, labels=[], dep=[], alignment_order=('e', 'f'), limit=0):
        self.limit = limit
        self.e_file = codecs.open(e_file, "r", "utf-8")
        self.f_files = [codecs.open(f, "r", "utf-8") for f in f_files]
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
        for f in self.f_files:
            f.seek(0)

    def get_f(self):
        results = []
        for f in self.f_files:
            f_split = f.readline().strip().split()
            try:
                toks, heads = zip(*[t.split("_") for t in f_split])
            except ValueError:
                toks = f_split
                heads = []
            results.append((toks, map(int, heads)))
        return results

    def get_als(self):
        return [map(self.order, re.findall("(\d+)-(\d+)", a.readline().strip())) for a in self.alignment_files]


    def __iter_source_dep(self):
        self.reset()

        e_line = self.e_file.readline()


        c = 0
        while(e_line):
            c += 1
            e_toks = e_line.strip().split()
            f_pairs = self.get_f()
            als = self.get_als()
            if self.limit and c > self.limit:
                break
            yield e_toks, f_pairs, als
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
    S = len(sure)
    A = len(als)
    a_s = 0
    a_p = 0
    for al in als:
        al_correct = False
        if al in sure:
            al_correct = True
            correct.add(al)
            sure.remove(al)
            a_s += 1
            a_p += 1
        if al in probable:
            al_correct = True
            correct.add(al)
            probable.remove(al)
            a_p += 1
        if not al_correct:
            wrong.add(al)

    recall = float(a_s) / S
    precision = float(a_p) / A
    fmeasure = (2 * precision * recall) / (precision + recall)
    aer = 1 - ((a_s + a_p) / float(A + S))
    eval = (precision, recall, fmeasure, aer)
    eval = " ".join(map("{:6.4f}".format, [round(x, 4) for x in eval]))
    return correct, wrong, sure, probable, eval

def get_null(J, a):
    _, f = zip(*a)
    aligned = set(f)
    null = []
    for j in xrange(J):
        if j not in aligned:
            null.append(j)

    return null



def escape(string):
    string = re.sub(r"\\", "\\textbackslash", string)
    string = re.sub(r"\^", "\\^{}", string)
    return re.sub(ESCAPE, r"\\\1", string)

def make_node(tok, pos, side, null=False):
    null_color = ""
    if null:
        null_color = "blue, "
    tok = escape(tok)
    return "\\node (%s%d) [word, %smy right of=%s%d] {%s};\n" % (side, pos, null_color, side, pos-1, tok)

def make_link(e_pos, f_pos, typ="link"):
    return "\\draw [%s] (f%d) -- (e%d);\n" % (typ, f_pos, e_pos)

def make_deps(heads):
    string = ""
    if len(heads) <= 1:
        return string
    deg_unit = 70.0 / (len(heads)-1)

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
    nulls = get_null(len(f), a)
    if heads:
        root = heads.index(-1)
    null_color = ""
    if 0 in nulls:
        null_color = ", blue"
    string = "\\node (f0) [word%s] {%s};\n" % (null_color, f[0])
    string += "\\node (e0) [word, my below of=f0] {%s};\n" % e[0]
    for i, f_tok in enumerate(f[1:]):
        string += make_node(f_tok, i+1, "f", null=i+1 in nulls)
    for i, e_tok in enumerate(e[1:]):
        string += make_node(e_tok, i+1, "e")
    eval = ()
    if gold:
        sure, probable = gold
        correct, wrong, sure, probable, eval = analyze_alignments(a, sure, probable)
        for coll, typ in [(correct, "correct"), (wrong, "wrong"), (sure, "sure"), (probable, "possible")]:
            for e_pos, f_pos in coll:
                string += make_link(e_pos, f_pos, typ=typ)
    else:
        for e_pos, f_pos in a:
            string += make_link(e_pos, f_pos)
    if heads:
        string += "\\node (root) [word, my above of=f%s] {ROOT};\n" % root
        string += make_deps(heads)


    return string, eval

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

def visualize_all(corpus, file_name, max_sent_length=0, gold_alignments=None):
    current = os.getcwd()
    temp = tempfile.mkdtemp()
    os.chdir(temp)

    labels = corpus.labels
    dep = corpus.dep

    all_files = []
    for i, (e, f_pairs, als) in enumerate(corpus):
        files = []
        if max_sent_length and max(len(e), len(f_pairs[0][0])) > max_sent_length:
            continue
        gold = []
        if gold_alignments:
            gold = gold_alignments.get_gold_alignments(i+1)
        for a_i, a in enumerate(als):
            f_i = a_i
            if a_i >= len(f_pairs):
                f_i = 0
            f, f_heads = f_pairs[f_i]
            tmp_heads = []
            if dep[a_i]:
                tmp_heads = f_heads
            code, eval = visualize(e,f,a, heads=tmp_heads, gold=gold)
            ftemp_name = "aligned_%d.%d.pdf" % (i, a_i)
            make_image(code, ftemp_name, label="Sent. " + str(i+1) + ", %s : %s" % (labels[a_i], eval))
            files.append(ftemp_name)

        if len(files) > 1:
            merged_file_name = "merged.%d.pdf" % i
            # pdfjam aligned_0.pdf aligned_1.pdf --nup 1x2 --outfile test.pdf
            proc = subprocess.Popen(["pdfjam"] + files + ["--nup", "1x" + str(len(als)), "--fitpaper",
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
    arg_parser.add_argument("-out")
    arg_parser.add_argument("-e_file")
    arg_parser.add_argument("-f_files", nargs="+")
    arg_parser.add_argument("-a_files", nargs="+")
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


    corpus = DataReader(args.e_file, args.f_files, args.a_files, labels=labels, dep=dep, alignment_order=alignment_order, limit=args.limit)

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