from Corpus_Reader import Corpus_Reader
import codecs
import subprocess
import tempfile
import shutil
import os

HEADER = R"""\documentclass[class=minimal,border=0pt]{standalone}
\usepackage[utf8]{inputenc}
\usepackage{tikz}
\usepackage{color}
\usetikzlibrary{shapes.geometric, arrows, positioning}
\tikzstyle{word} = [text centered]
\tikzstyle{link} = [thick]
\begin{document}
\tikzstyle{my below of} = [below=of #1.south]
\tikzstyle{my right of} = [right=of #1.east]
\tikzstyle{my left of} = [left=of #1.west]
\tikzstyle{my above of} = [above=of #1.north]
\begin{tikzpicture}[auto, node distance=.7cm,>=latex']"""

FOOTER = r"""\end{tikzpicture}
\end{document}"""

def make_node(tok, pos, side):
    return "\\node (%s%d) [word, my right of=%s%d] {%s};\n" % (side, pos, side, pos-1, tok)

def make_link(e_pos, f_pos):
    return "\\draw [link] (f%d) -- (e%d);\n" % (f_pos, e_pos)

def visualize(e,f,a):
    string = "\\node (f0) [word] {%s};\n" % f[0]
    string += "\\node (e0) [word, my below of=f0] {%s};\n" % e[0]
    for i, f_tok in enumerate(f[1:]):
        string += make_node(f_tok, i+1, "f")
    for i, e_tok in enumerate(e[1:]):
        string += make_node(e_tok, i+1, "e")
    for e_pos, f_pos in a:
        string += make_link(e_pos, f_pos)

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

def visualize_all(corpus, file_name, max_sent_length=0):
    current = os.getcwd()
    temp = tempfile.mkdtemp()
    os.chdir(temp)

    files = []
    for i, (a,e,f) in enumerate(corpus):
        if max_sent_length and max(len(e), len(f)) > max_sent_length:
            continue
        code = visualize(e,f,a)
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

    corpus = Corpus_Reader(args.e_file, args.f_file, args.a_file, alignment_order=alignment_order, limit=limit)

    visualize_all(corpus, args.out, max_sent_length=args.max_length)
