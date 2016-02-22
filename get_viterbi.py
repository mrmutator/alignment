from Corpus_Reader import Corpus_Reader
import cPickle as pickle
import multiprocessing as mp
import numpy as np
import argparse

def get_all_viterbi_alignments(data, trans_prob, al_prob, p_0, results, group):
    all_alignments = []
    for e_toks, f_toks in data:
        J = len(f_toks)
        I = len(e_toks)
        chart = np.zeros((J, 2*I))
        best = np.zeros((J, 2*I))
        # initialize
        for i, e_tok in enumerate(e_toks):
            chart[0][i] = trans_prob[(e_tok,f_toks[0])] * al_prob[((None, I),i)]
        for i in range(I, I*2):
            chart[0][i] = trans_prob[(0,f_toks[0])] * p_0

        # compute chart
        for j, f_tok in enumerate(f_toks[1:]):
            j= j+1
            for i, e_tok in enumerate(e_toks + [0]*I):
                values = []
                for i_p in range(2*I):
                    if i >= I:
                        if i - I == i_p or i == i_p: # i is NULL
                            values.append(chart[j-1][i_p] * p_0)
                        else:
                            values.append(0)
                    else:
                        if i_p < I:
                            values.append(chart[j-1][i_p]*al_prob[(I,i-i_p)])
                        else:
                            values.append(chart[j-1][i_p]*al_prob[(I,i-i_p + I)])
                best_i = np.argmax(values)
                chart[j][i] = values[best_i] * trans_prob[(e_tok,f_tok)]
                best[j][i] = best_i

        # get max path
        best_path = []
        best_end = np.argmax(chart[J-1])
        best_path.append(best_end)
        for j in reversed(range(1, J)):
            best_end = best[j][best_end]
            best_path.append(best_end)

        best_path = list(reversed(best_path))
        alignments = [(int(best_path[j]), j) for j in range(J) if int(best_path[j]) < I]
        all_alignments. append(alignments)
    results.put((group, all_alignments))

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-e", required=True)
arg_parser.add_argument("-f", required=True)
arg_parser.add_argument("-prms", required=True)
arg_parser.add_argument("-num_workers", required=True, type=int)
arg_parser.add_argument("-out_file", required=True)
arg_parser.add_argument("-p_0", required=False, default=0.2, type=float)

args = arg_parser.parse_args()

corpus = Corpus_Reader(args.e, args.f)
trans_params, al_params = pickle.load(open(args.prms, "rb"))
num_workers = args.num_workers
p_0 = args.p_0

corpus = list(corpus)
n= int(np.ceil(float(len(corpus)) / num_workers))
data = [corpus[i:i+n] for i in range(0, len(corpus), n)]

results = mp.Queue()

processes = [mp.Process(target=get_all_viterbi_alignments, args=(data[i], trans_params, al_params, p_0, results, i)) for i in xrange(num_workers)]
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