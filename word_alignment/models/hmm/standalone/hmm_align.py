from word_alignment.utils.Corpus_Reader import Corpus_Reader, GIZA_Reader
from parameters import prepare_data
import hmm
import numpy as np
from collections import Counter, defaultdict
import argparse
import multiprocessing as mp
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()


def load_params(params, file_name, p_0=0.2, alpha=0.0):
    lengths = set()
    start_params = dict()
    jumps = dict()
    infile = open(file_name, "r")
    for line in infile:
        key, value = line.strip().split("\t")
        if key.startswith("b/"):
            els = key.split("/")
            I = int(els[1])
            lengths.add(I)
            start_params[I, int(els[2])] = float(value)
        elif key.startswith("j/"):
            els = key.split("/")
            jumps[int(els[1])] = float(value)
        elif key.startswith("t/"):
            params[key] = float(value)
        else:
            raise Exception("Should not happen.")
    infile.close()
    params["I"] = lengths
    for I in lengths:
        tmp_prob = np.zeros((I, I))
        for i_p in xrange(I):
            norm = np.sum([ jumps[i_pp - i_p] for i_pp in xrange(I)]) + p_0
            tmp_prob[i_p, :] = np.array([((jumps[i-i_p] / norm) * (1-alpha)) + (alpha * (1.0/I))  for i in xrange(I)])
        params["d/"+str(I)] = tmp_prob

        tmp2_prob = np.zeros(I)
        for i in xrange(I):
            tmp2_prob[i] = start_params[I, i]
        params["s/" + str(I)] = tmp2_prob



class Corpus_Buffer(object):

    def __init__(self, e_file, f_file, strings="str", buffer_size = 20):
        self.buffer_size = buffer_size
        self.corpus = Corpus_Reader(e_file, f_file, strings=strings)
        self.limit = 0

    def __iter__(self):
        self.corpus.reset()
        buffer = []
        c = 0
        for el in self.corpus:
            buffer.append(el)
            c += 1
            if c == self.limit:
                break
            if c == self.buffer_size:
                yield buffer
                buffer = []
                c = 0
        if c > 0:
            yield buffer

def get_all_viterbi_alignments(corpus, params, results_queue, worker_num):
    all_alignments = []
    for e_toks, f_toks in corpus:
        J = len(f_toks)
        I = len(e_toks)
        al_prob = params["d/"+str(I)]
        start_prob = params["s/"+str(I)]
        trans_params = set(["t/" + e_tok + "/" + f_tok for f_tok in f_toks for e_tok in e_toks + ["0"]])
        trans_params = {k: params.get(k, 0.00000001) for k in trans_params}

        chart = np.zeros((J, 2*I))
        best = np.zeros((J, 2*I))
        # initialize
        for i, e_tok in enumerate(e_toks):
            chart[0][i] = trans_params["t/" + e_tok + "/" + f_toks[0]] * start_prob[i]
        for i in range(I, I*2):
            chart[0][i] = trans_params["t/0" + "/" + f_toks[0]] * p_0

        # compute chart
        for j, f_tok in enumerate(f_toks[1:]):
            j= j+1
            for i, e_tok in enumerate(e_toks + ["0"]*I):
                values = []
                for i_p in range(2*I):
                    if i >= I:
                        if i - I == i_p or i == i_p: # i is NULL
                            values.append(chart[j-1][i_p] * p_0)
                        else:
                            values.append(0)
                    else:
                        if i_p < I:
                            values.append(chart[j-1][i_p]*al_prob[i_p][i])
                        else:
                            values.append(chart[j-1][i_p]*al_prob[i_p-I][i])
                best_i = np.argmax(values)
                chart[j][i] = values[best_i] * trans_params["t/" + e_tok + "/" + f_tok]
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
    result_queue.put((worker_num, all_alignments))

def update_parameters(counts, pnames, file_name):
    outfile = open(file_name, "wb")
    # update parameters
    for (e,f), count in counts[0].items():
        name = "t/" + e + "/" + f
        if name in pnames:
            value = count / counts[1][e]
            outfile.write(name + "\t" + str(value) + "\n")

    jmp_prob = defaultdict(int)
    for (i_p, i), count in counts[3].items():
        jmp_prob[i-i_p] += count / counts[2][i_p]

    for jmp in jmp_prob:
        outfile.write("j/" + str(jmp) + "\t" + str(jmp_prob[jmp]) + "\n")


    for (I, i), count in counts[4].items():
        value = count / counts[5][I]
        outfile.write("b/" + str(I) + "/" + str(i) + "\t" + str(value) + "\n")

    outfile.close()

def train_iteration(corpus, queue):
    # set all counts to zero
    lex_counts = Counter() #(e,f)
    lex_norm = Counter() # e
    start_counts = Counter() # (I, i)
    start_norm = Counter() # I
    al_counts = Counter() # (i_p, i)
    al_norm = Counter() # (i_p)
    ll = 0
    for e_toks, f_toks in corpus:
        I = len(e_toks)
        I_double = 2 * I
        J = len(f_toks)

        trans_params = set(["t/" + e_tok + "/" + f_tok for f_tok in f_toks for e_tok in e_toks + ["0"]])

        # get parameters
        current_params = {k: params.get(k, 0.00000001) for k in ["s/" + str(I), "d/" + str(I)] + list(trans_params)}
        trans_params = [tp for tp in trans_params if current_params[tp] > 0.00000001]

        start_prob = np.hstack((current_params["s/" + str(I)], np.ones(I) * (p_0/I)))


        tmp = np.hstack((current_params["d/" + str(I)], np.identity(I)))
        dist_mat = np.vstack((tmp, tmp))

        # initialize alphas, betas and scaling coefficients

        alphas = np.zeros((J, I_double))
        betas = np.ones((J, I_double))
        scale_coeffs = np.zeros(J)

        trans_mat = np.zeros((I_double, J))
        for j, f_tok in enumerate(f_toks):
            for i, e_tok in enumerate(e_toks + ["0"] * I):
                trans_mat[i, j] = current_params["t/"+ e_tok + "/" + f_tok]


        hmm.forward(J, I_double, start_prob, dist_mat, trans_mat, alphas, scale_coeffs)
        hmm.backward(J, I_double, dist_mat, trans_mat, betas, scale_coeffs)

        gammas = np.multiply(alphas, betas)

        # update counts

        # add start counts and counts for lex f_0
        f_0 = f_toks[0]
        for i, e_tok in enumerate(e_toks):
            start_counts[(I, i)] += gammas[0][i]
            if "t/" + e_tok + "/" + f_0 in trans_params:
                lex_counts[(e_tok, f_0)] += gammas[0][i]
                lex_norm[e_tok] += gammas[0][i]
        start_norm[I] += 1
        if "t/0/" + f_0 in trans_params:
            zero_sum = np.sum(gammas[0][I:])
            lex_counts[("0", f_0)] += zero_sum
            lex_norm["0"] += zero_sum

        for j_p, f_tok in enumerate(f_toks[1:]):
            j = j_p + 1
            t_f_e = np.array([current_params["t/" + e_tok + "/" + f_toks[j]] for e_tok in e_toks + ["0"]*I]) # array of t(f_j|e) for all e
            beta_t_j_i = np.multiply(betas[j], t_f_e)
            alpha_j_p = alphas[j_p]
            if "t/0/"+f_tok in trans_params:
                gammas_0_j = np.sum(gammas[j][I:])
                lex_counts[("0", f_tok)] += gammas_0_j
                lex_norm["0"] += gammas_0_j
            for i, e_tok in enumerate(e_toks):
                if "t/" + e_tok + "/" + f_tok in trans_params:
                    lex_counts[(e_tok, f_tok)] += gammas[j][i]
                    lex_norm[e_tok] += gammas[j][i]
                for i_p in range(I_double):
                    if i_p < I:
                        al_prob_ip = current_params["d/"+ str(I)][i_p]
                        actual_i_p = i_p
                    else:
                        al_prob_ip = current_params["d/" + str(I)][i_p-I]
                        actual_i_p = i_p - I
                    xi = (al_prob_ip[i]  * alpha_j_p[i_p] * beta_t_j_i[i]) / scale_coeffs[j]
                    al_counts[(actual_i_p, i)] += xi
                    al_norm[actual_i_p] += gammas[j_p][i_p]


        ll += np.sum(np.log(scale_coeffs))

    queue.put((lex_counts, lex_norm, al_norm, al_counts, start_counts, start_norm, ll))

def write_alignments(al_groups, out_file_name):
    outfile = open(out_file_name, "w")
    for al_group in al_groups:
        for als in al_group[1]:
            als = [str(al[0]) + "-" + str(al[1]) for al in als]
            outfile.write(" ".join(als) + "\n")
    outfile.close()

def aggregate_counts(queue, ll, num_work, pnames):
    # total = [Counter(), Counter(), Counter(), Counter(), Counter(), Counter()]
    # total_ll = 0
    initial = queue.get()
    total = initial[:-1]
    total_ll = initial[-1]
    for _ in xrange(num_work-1):
        counts = queue.get()
        for i, c in enumerate(counts[:-1]):
            total[i].update(c)
        total_ll += counts[-1]
    logger.info("U-Step iteration %d" % (it+1))
    update_parameters(total,pnames, "params.p")

    ll["ll"] = total_ll

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-giza_file", required=True)
    arg_parser.add_argument("-t_file", required=True)
    arg_parser.add_argument("-output_vocab_file", required=False, default="")
    arg_parser.add_argument("-num_workers", required=True, type=int, default=1)
    arg_parser.add_argument("-file_prefix", required=False, default="tmp")
    arg_parser.add_argument("-e_voc", required=False, default="")
    arg_parser.add_argument("-f_voc", required=False, default="")
    arg_parser.add_argument("-alpha", required=False, default=0.0, type=float)
    arg_parser.add_argument("-p_0", required=False, default=0.2, type=float)
    arg_parser.add_argument("-num_iterations", required=False, default=5, type=int)
    arg_parser.add_argument("-buffer_size", required=False, default=20, type=int)
    arg_parser.add_argument("-viterbi_limit", required=False, default=0, type=int)


    args = arg_parser.parse_args()

    p_0 = args.p_0


    corpus = GIZA_Reader(args.giza_file)
    corpus_size = prepare_data(corpus, p_0=args.p_0, t_file=args.t_file, num_workers=args.num_workers,
                                                        file_prefix=args.file_prefix,
                                                        output_vocab_file=args.output_vocab_file, random=False)

    logger.info("Parameters ready.")
    corpus_buffer = Corpus_Buffer(e_file = args.file_prefix + ".e", f_file=args.file_prefix + ".f", buffer_size=args.buffer_size)

    num_work = int(np.ceil(float(corpus_size) / args.buffer_size))



    for it in xrange(args.num_iterations):
        alpha = args.alpha
        if it == 0:
            alpha = 1.0 # uniform jump parameters for initialization

        logger.info("E-Step iteration %d" % (it+1))
        update_queue = mp.Queue()
        process_queue = mp.Queue()
        manager = mp.Manager()
        ll = manager.dict()
        params = manager.dict()
        def worker_wrapper(process_queue):
            while True:
                buffer = process_queue.get()
                train_iteration(buffer, update_queue)


        pool = mp.Pool(args.num_workers-1,worker_wrapper, (process_queue,))
        load_params(params, "params.p", p_0=p_0, alpha=alpha)
        pnames = set(params.keys())
        updater = mp.Process(target=aggregate_counts, args=(update_queue, ll, num_work, pnames))
        updater.start()

        for buffer in corpus_buffer:
            process_queue.put(buffer)

        updater.join()

        total_ll = ll["ll"]

        del params
        del update_queue
        del process_queue
        del manager
        del updater
        del pool
        logger.info("Likelihood it. %d: %d" % (it+1, total_ll))

    # Viterbi
        if args.viterbi_limit >= 0:
            logger.info("Aligning test part.")
            corpus_buffer.limit = args.viterbi_limit
            manager = mp.Manager()
            params = manager.dict()
            process_queue = mp.Queue()
            result_queue = mp.Queue()
            def viterbi_wrapper(process_queue):
                while True:
                    i, buffer = process_queue.get()
                    get_all_viterbi_alignments(buffer, params, result_queue, i)
            pool = mp.Pool(max(args.num_workers,4), viterbi_wrapper, (process_queue,))
            load_params(params, "params.p", p_0=p_0, alpha=alpha)

            c = 0
            for i, buffer in enumerate(corpus_buffer):
                process_queue.put((i, buffer))
                c += 1
            results = []
            for _ in xrange(c):
                results.append(result_queue.get())

            results = sorted(results, key=lambda t: t[0])
            write_alignments(results, args.file_prefix + ".aligned." + str(it+1))
            corpus_buffer.limit = 0
            del params
            del pool
            del process_queue
            del result_queue
            del manager
    logger.info("Done.")

