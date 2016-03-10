from word_alignment.utils.Corpus_Reader import Corpus_Reader
from parameters import prepare_data
import hmm
import numpy as np
from collections import Counter, defaultdict
import argparse
import multiprocessing as mp

class Corpus_Buffer(object):

    def __init__(self, e_file, f_file, strings=False, buffer_size = 20):
        self.buffer_size = buffer_size
        self.corpus = Corpus_Reader(e_file, f_file, strings=strings)

    def __iter__(self):
        self.corpus.reset()
        buffer = []
        c = 0
        for el in self.corpus:
            buffer.append(el)
            c += 1
            if c == self.buffer_size:
                yield buffer
                buffer = []
                c = 0
        if c > 0:
            yield buffer

def update_counts(counts, trans_prob, dist_params, start_params, alpha=0, p_0 = 0.2):

    # update parameters
    for (e,f), count in counts[0].items():
        trans_prob[(e,f)] = count / counts[1][e]


    jmp_prob = defaultdict(int)
    for (i_p, i), count in counts[3].items():
        jmp_prob[i-i_p] += count / counts[2][i_p]

    dist_params = dict()
    for I in dist_params:
        tmp_prob = np.zeros((I, I))
        for i_p in xrange(I):
            norm = np.sum([jmp_prob[i_pp - i_p] for i_pp in xrange(I)]) + p_0
            tmp_prob[i_p, :] = np.array([((jmp_prob[i-i_p] / norm) * (1-alpha)) + (alpha * (1.0/I))  for i in xrange(I)])
        dist_params[I] = tmp_prob


    start_prob = dict()
    for (I, i), count in counts[4].items():
        start_prob[(I, i)] = count / counts[5][I]

    start_params = dict()
    for I in start_params:
        tmp_prob = np.zeros(I)
        for i in xrange(I):
            tmp_prob[i] = start_prob[(I, i)]
        start_params[I] = tmp_prob


def train_iteration(corpus):
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
        start_prob = np.hstack((start_params[I], np.ones(I) * (p_0/I)))

        tmp = np.hstack((dist_params[I], np.identity(I)))
        dist_mat = np.vstack((tmp, tmp))

        # initialize alphas, betas and scaling coefficients

        alphas = np.zeros((J, I_double))
        betas = np.ones((J, I_double))
        scale_coeffs = np.zeros(J)

        trans_mat = np.zeros((I_double, J))
        for j, f_tok in enumerate(f_toks):
            for i, e_tok in enumerate(e_toks + [0] * I):
                trans_mat[i, j] = trans_prob[(e_tok,f_tok)]

        hmm.forward(J, I_double, start_prob, dist_mat, trans_mat, alphas, scale_coeffs)
        hmm.backward(J, I_double, dist_mat, trans_mat, betas, scale_coeffs)

        gammas = np.multiply(alphas, betas)

        # update counts

        # add start counts and counts for lex f_0
        f_0 = f_toks[0]
        for i, e_tok in enumerate(e_toks + [0]*I):
            start_counts[(I, i)] += gammas[0][i]
            lex_counts[(e_tok, f_0)] += gammas[0][i]
            lex_norm[e_tok] += gammas[0][i]
        start_norm[I] += 1

        for j_p, f_tok in enumerate(f_toks[1:]):
            j = j_p + 1
            t_f_e = np.array([trans_prob[(e_tok,f_toks[j])] for e_tok in e_toks + [0]*I]) # array of t(f_j|e) for all e
            beta_t_j_i = np.multiply(betas[j], t_f_e)
            alpha_j_p = alphas[j_p]
            gammas_0_j = np.sum(gammas[j][I:])
            lex_counts[(0, f_tok)] += gammas_0_j
            lex_norm[0] += gammas_0_j
            for i, e_tok in enumerate(e_toks):
                lex_counts[(e_tok, f_tok)] += gammas[j][i]
                lex_norm[e_tok] += gammas[j][i]
                for i_p in range(I_double):
                    if i_p < I:
                        al_prob_ip = dist_params[I][i_p]
                        actual_i_p = i_p
                    else:
                        al_prob_ip = dist_params[I][i_p-I]
                        actual_i_p = i_p - I
                    xi = (al_prob_ip[i]  * alpha_j_p[i_p] * beta_t_j_i[i]) / scale_coeffs[j]
                    al_counts[(actual_i_p, i)] += xi
                    al_norm[actual_i_p] += gammas[j_p][i_p]


        ll += np.sum(np.log(scale_coeffs))

    return (lex_counts, lex_norm, al_norm, al_counts, start_counts, start_norm, ll)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-e", required=True)
    arg_parser.add_argument("-f", required=True)
    arg_parser.add_argument("-limit", required=False, type=int, default=0)
    arg_parser.add_argument("-output_vocab_file", required=False, default="")
    arg_parser.add_argument("-num_workers", required=True, type=int, default=1)
    arg_parser.add_argument("-file_prefix", required=False, default="tmp")
    arg_parser.add_argument("-t_file", required=False, default="")
    arg_parser.add_argument("-e_voc", required=False, default="")
    arg_parser.add_argument("-f_voc", required=False, default="")
    arg_parser.add_argument("-alpha", required=False, default=0.0, type=float)
    arg_parser.add_argument("-p_0", required=False, default=0.2, type=float)
    arg_parser.add_argument("-num_iterations", required=False, default=5, type=int)
    arg_parser.add_argument("-buffer_size", required=False, default=20, type=int)


    args = arg_parser.parse_args()

    p_0 = args.p_0
    alpha = args.alpha

    corpus = Corpus_Reader(args.e, args.f, limit=args.limit, strings=True)

    trans_prob, dist_params, start_params = prepare_data(corpus, e_voc=args.e_voc, f_voc=args.f_voc, alpha=args.alpha,
                                                        p_0=args.p_0, t_file=args.t_file, num_workers=args.num_workers,
                                                        file_prefix=args.file_prefix,
                                                        output_vocab_file=args.output_vocab_file)

    print "Parameters ready."
    corpus_buffer = Corpus_Buffer(e_file = args.file_prefix + ".e", f_file=args.file_prefix + ".f", buffer_size=args.buffer_size)
    for it in xrange(args.num_iterations):
        print "Iteration " + str(it+1)
        pool = mp.Pool(processes=args.num_workers)
        results = pool.map(train_iteration, corpus_buffer)

        initial = results[0]
        total = initial[:-1]
        total_ll = initial[-1]

        for counts in results:
            for i, c in enumerate(counts[:-1]):
                total[i].update(c)
            total_ll += counts[-1]

        print total_ll
        update_counts(total, trans_prob, dist_params, start_params)
