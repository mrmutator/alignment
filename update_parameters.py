import cPickle as pickle
from collections import defaultdict, Counter
import argparse
import glob
import numpy as np

def load_params(param_file):
    trans_params, jump_params, start_params = pickle.load(open(param_file, "rb"))
    return trans_params, jump_params, start_params

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-dir", required=True)
arg_parser.add_argument("-alpha", required=False, default=0.0, type=float)
args = arg_parser.parse_args()


exp_files = glob.glob(args.dir.rstrip("/") + "/*.counts")
param_files = glob.glob(args.dir.rstrip("/") + "/*.prms")

alpha = args.alpha

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
    outfile.write("Log-Likelihood: " + str(total_ll))

trans_prob = dict()
jmp_prob = defaultdict(int)
start_prob = dict()
al_prob = dict()

# update parameters

for (e,f), count in total['counts_e_f'].items():
    trans_prob[(e,f)] = count / total['counts_e'][e]


for (i, i_p), count in total['xi_sums'].items():
    jmp_prob[i-i_p] += count / total['gamma_sums'][i_p]


for (I, i), count in total['pi_counts'].items():
    start_prob[(I, i)] = count / total['pi_denom'][I]


for f in param_files:
    trans_params, al_params, start_params  = load_params(f)
    for k in trans_params:
        trans_params[k] = trans_prob[k]
    for k in al_params:
        if k not in al_prob:
            tmp_prob = dict()
            for i_p in xrange(k): # k==I
                norm = np.sum([ jmp_prob[i_pp - i_p] for i_pp in xrange(k)])
                tmp_prob[i_p] = {i: ((jmp_prob[i-i_p] / norm) * (1-alpha)) + (alpha * (1.0/k))  for i in xrange(k)}
            al_prob[k] = tmp_prob
        al_params[k] = al_prob[k]

    for k in start_params:
        start_params[k] = start_prob[k]

    pickle.dump((trans_params, al_params, start_params), open(f +".u", "wb"))