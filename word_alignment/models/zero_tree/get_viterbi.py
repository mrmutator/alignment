from word_alignment.utils.Corpus_Reader import Corpus_Reader
import cPickle as pickle
import multiprocessing as mp
import numpy as np
import argparse

def get_all_viterbi_alignments(data, trans_prob, al_prob, results, group):
    all_alignments = []
    for e_toks, f_pairs in data:
        J = len(f_pairs)
        I = len(e_toks)

        alignments = []
        for j, (f_tok, f_head) in enumerate(f_pairs):
            probs_i = []
            for i, e_tok in enumerate([0] + e_toks):
                probs_i.append(trans_prob[(e_tok, f_tok)]*al_prob[(I, J, f_head, j), i])
            best = np.argmax(probs_i)
            if best != 0:
                alignments.append((best-1, j))

        all_alignments.append(alignments)
    results.put((group, all_alignments))

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-e", required=True)
arg_parser.add_argument("-f", required=True)
arg_parser.add_argument("-prms", required=True)
arg_parser.add_argument("-num_workers", required=True, type=int)
arg_parser.add_argument("-out_file", required=True)

args = arg_parser.parse_args()

corpus = Corpus_Reader(args.e, args.f, source_dep=True)
trans_params, al_params = pickle.load(open(args.prms, "rb"))
num_workers = args.num_workers

corpus = list(corpus)
n= int(np.ceil(float(len(corpus)) / num_workers))
data = [corpus[i:i+n] for i in range(0, len(corpus), n)]
num_workers = len(data)

results = mp.Queue()

processes = [mp.Process(target=get_all_viterbi_alignments, args=(data[i], trans_params, al_params, results, i)) for i in xrange(num_workers)]
for p in processes:
    p.start()

alignments = []

for p in processes:
    als = results.get()
    alignments.append(als)

for p in processes:
    a = p.join()


outfile = open(args.out_file, "w")
for group in sorted(alignments, key=lambda t: t[0]):
    for als in group[1]:
        als = [str(al[0]) + "-" + str(al[1]) for al in als]
        outfile.write(" ".join(als) + "\n")
outfile.close()