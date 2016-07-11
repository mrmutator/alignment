from collections import defaultdict, Counter
import argparse
import glob
import multiprocessing as mp
import re
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()


def update_count_file(file_name, total):
    with open(file_name, "r") as infile:
        ll = float(infile.readline().strip().split("\t")[1])
        total[6] += ll

        for line in infile:
            count_i, k, v = line.strip().split("\t")
            count_i = int(count_i)
            v = float(v)
            k = k.split(" ")
            if len(k) == 1:
                k = int(k[0])
            else:
                k = tuple(map(int, k))
            total[count_i][k] += v


def write_param_file(count_file_name, normalized_counts):
    param_file_name = re.sub(r"counts\.(\d+)$", r"params.\1", count_file_name)
    lengths_I = set()
    pos_jumps = defaultdict(set)
    with open(param_file_name, "w") as outfile:
        with open(count_file_name, "r") as infile:
            infile.readline()
            for line in infile:
                count_i, k, _ = line.strip().split("\t")
                count_i = int(count_i)
                if count_i == 0:
                    k_str = k.split(" ")
                    k_int = (int(k_str[0]), int(k_str[1]))
                    if k_int in normalized_counts["trans_prob"]:
                        value = str(normalized_counts["trans_prob"][k_int])
                        outfile.write(" ".join(["t", k_str[0], k_str[1], value]) + "\n")
                elif count_i == 5:
                    k = int(k)
                    lengths_I.add(k)
                elif count_i == 2:
                    k_str = k.split(" ")
                    k_int = (int(k_str[0]), int(k_str[1]), int(k_str[2]))
                    jmp = k_int[2]-k_int[1]
                    pos_jumps[k_int[0]].add(jmp)

        for I in lengths_I:
            for i in xrange(I):
                value = normalized_counts["start_prob"][(I, i)]
                key_str = ["s"] + map(str, [I, i, value])
                outfile.write(" ".join(key_str) + "\n")

        for p in pos_jumps:
            max_I = max(pos_jumps[p])+1
            for jmp in xrange(-max_I + 1, max_I):
                value = normalized_counts["jmp_prob"][p, jmp]
                key_str = ["j"] + map(str, [p, jmp, value])
                outfile.write(" ".join(key_str) + "\n")


def normalize_trans(queue):
    trans_prob = dict()
    for (e, f), count in lex_counts.iteritems():
        v = count / lex_norm[e]
        if v > 0.00000001:
            trans_prob[(e, f)] = v
    queue.put(("trans_prob", trans_prob))


def normalize_jumps(queue):
    jmp_prob = defaultdict(int)
    for (p, i_p, i), count in al_counts.iteritems():
        jmp_prob[p, i - i_p] += count / al_norm[p, i_p]
    queue.put(("jmp_prob", jmp_prob))


def normalize_start(queue):
    start_prob = dict()
    for (I, i), count in start_counts.iteritems():
        start_prob[(I, i)] = count / start_norm[I]
    queue.put(("start_prob", start_prob))


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-dir", required=True)
args = arg_parser.parse_args()

exp_files = glob.glob(args.dir.rstrip("/") + "/*.counts.*")

# types = ["lex_counts", "lex_norm", "al_counts", "al_norm", "start_counts", "start_norm", "ll"]
total = [Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), 0.0]

logger.info("Aggregating counts.")
for f in exp_files:
    update_count_file(f, total)

lex_counts, lex_norm, al_counts, al_norm, start_counts, start_norm, ll = total

logger.info("Normalizing counts.")
results = mp.Queue()
processes = [mp.Process(target=x, args=(results,)) for x in
             [normalize_start, normalize_jumps, normalize_trans]]

for p in processes:
    p.start()

normalized_counts = dict()

for p in processes:
    name, data = results.get()
    normalized_counts[name] = data
for p in processes:
    a = p.join()

logger.info("Writing parameter files.")
for f in exp_files:
    write_param_file(f, normalized_counts)

logger.info("Log-likelihood before update: %s" % ll)
with open("log_likelihood", "w") as outfile:
    outfile.write("Log-Likelihood: " + str(ll) + "\n")
