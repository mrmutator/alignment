import cPickle as pickle
from collections import defaultdict, Counter
import argparse
import glob
import numpy as np
import multiprocessing as mp

def load_params(p_list_file):
    trans_params = dict()
    lengths = set()
    infile = open(p_list_file, "r")
    for line in infile:
        k, v = line.strip().split("\t")
        if k == "t":
            tpl = v.split(" ")
            trans_params[(int(tpl[0]), int(tpl[1]))] = 0
        elif k == "I":
            lengths.add(int(v))

    return trans_params, lengths

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-dir", required=True)
arg_parser.add_argument("-alpha", required=False, default=0.0, type=float)
arg_parser.add_argument("-num_workers", required=False, default=1, type=int)
arg_parser.add_argument("-p_0", required=False, default=0.2, type=float)
args = arg_parser.parse_args()


exp_files = glob.glob(args.dir.rstrip("/") + "/*.counts")
param_files = glob.glob(args.dir.rstrip("/") + "/*.plist")

alpha = args.alpha
p_0 = args.p_0

total = defaultdict(Counter)

total_ll = 0

for f in exp_files:
    expectations = pickle.load(open(f, "rb"))

    total_ll += expectations["ll"]
    del expectations["ll"]

    for count, d in expectations.iteritems():
        total[count].update(d)

print "LL before update: ", total_ll
with open("log_likelihood", "w") as outfile:
    outfile.write("Log-Likelihood: " + str(total_ll) + "\n")

# update parameters

def normalize_trans(queue):
    trans_prob = dict()
    for (e,f), count in total['lex_counts'].items():
        trans_prob[(e,f)] = count / total['lex_norm'][e]
    queue.put(("trans_prob", trans_prob))

def normalize_jumps(queue):
    jmp_prob = defaultdict(int)
    for (i_p, i), count in total['al_counts'].items():
        jmp_prob[i-i_p] += count / total['al_norm'][i_p]
    queue.put(("jmp_prob", jmp_prob))

def normalize_start(queue):
    start_prob = dict()
    for (I, i), count in total['start_counts'].items():
        start_prob[(I, i)] = count / total['start_norm'][I]
    queue.put(("start_prob", start_prob))


results = mp.Queue()

processes = [mp.Process(target=x, args=(results, )) for x in [normalize_start, normalize_jumps, normalize_trans]]
for p in processes:
    p.start()

normalized_counts = dict()

for p in processes:
    name, data = results.get()
    normalized_counts[name] = data
for p in processes:
    a = p.join()

manager = mp.Manager()
al_prob = manager.dict()
start_prob = manager.dict()

def update_worker(f):
    trans_params, lengths = load_params(f)
    dist_params = dict()
    start_params = dict()
    for k in trans_params:
        trans_params[k] = normalized_counts['trans_prob'][k]
    for I in lengths:
        if I not in al_prob:
            tmp_prob = np.zeros((I, I))
            for i_p in xrange(I):
                norm = np.sum([ normalized_counts['jmp_prob'][i_pp - i_p] for i_pp in xrange(I)]) + p_0
                tmp_prob[i_p, :] = np.array([((normalized_counts['jmp_prob'][i-i_p] / norm) * (1-alpha)) + (alpha * (1.0/I))  for i in xrange(I)])
            al_prob[I] = tmp_prob
        dist_params[I] = al_prob[I]

        if I not in start_prob:
            tmp_prob = np.zeros(I)
            for i in xrange(I):
                tmp_prob[i] = normalized_counts['start_prob'][(I, i)]

            start_prob[I] = tmp_prob
        start_params[I] = start_prob[I]


    pickle.dump((trans_params, dist_params, start_params), open(f[:-5] +"prms.u", "wb"))


pool = mp.Pool(processes=args.num_workers)
pool.map(update_worker, param_files)