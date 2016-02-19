import cPickle as pickle
from collections import defaultdict, Counter
import argparse
import glob

def load_params(param_file):
    trans_params, al_params = pickle.load(open(param_file, "rb"))
    return trans_params, al_params

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-dir", required=True)
args = arg_parser.parse_args()


exp_files = glob.glob(args.dir.rstrip("/") + "/*.counts")
param_files = glob.glob(args.dir.rstrip("/") + "/*.prms")


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
al_prob = dict()

# update parameters

for (e,f), count in total['counts_e_f'].items():
    trans_prob[(e,f)] = count / total['counts_e'][e]

jmp_dict = defaultdict(lambda: defaultdict(int))
for (i, i_p, I), count in total['xi_sums'].items():
    jmp_dict[I][i-i_p] += count / total['gamma_sums'][(i_p, I)]
for I_p in jmp_dict:
    norm_c = sum(jmp_dict[I_p].values())
    for jmp in jmp_dict[I_p]:
        al_prob[(I_p, jmp)] = jmp_dict[I_p][jmp] / norm_c

for (i, I), count in total['pi_counts'].items():
    al_prob[((None, I), i)] = count / total['pi_denom'][I]


for f in param_files:
    trans_params, al_params = load_params(f)
    for k in trans_params:
        trans_params[k] = trans_prob[k]
    for k in al_params:
        al_params[k] = al_prob[k]

    pickle.dump((trans_params, al_params), open(f +".u", "wb"))