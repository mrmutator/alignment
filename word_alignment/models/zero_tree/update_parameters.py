import cPickle as pickle
from collections import defaultdict, Counter
import argparse
import glob
import multiprocessing as mp

def load_params(param_file):
    trans_params, al_params = pickle.load(open(param_file, "rb"))
    return trans_params, al_params

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-dir", required=True)
arg_parser.add_argument("-alpha", required=False, default=0.0, type=float)
arg_parser.add_argument("-num_workers", required=False, default=1, type=int)
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
    outfile.write("Log-Likelihood: " + str(total_ll) + "\n")

# update parameters

def normalize_trans(queue):
    trans_prob = dict()
    for (e,f), count in total['lex_counts'].items():
        trans_prob[(e,f)] = count / total['lex_norm'][e]
    queue.put(("trans_prob", trans_prob))

def normalize_al(queue):
    al_prob = dict()
    for (I, J, f_head, j, i), count in total['al_counts'].items():
        al_prob[((I, J, f_head, j), i)] = count / total['al_norm'][(I, J, f_head, j)]
    queue.put(("al_prob", al_prob))


results = mp.Queue()

processes = [mp.Process(target=x, args=(results, )) for x in [normalize_al, normalize_trans]]
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

def update_worker(f):
    trans_params, al_params  = load_params(f)
    for k in trans_params:
        trans_params[k] = normalized_counts['trans_prob'][k]

    for k in al_params:
        al_params[k] = normalized_counts['al_prob'][k]

    pickle.dump((trans_params, al_params), open(f +".u", "wb"))


pool = mp.Pool(processes=args.num_workers)
pool.map(update_worker, param_files)