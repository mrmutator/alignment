from __future__ import division
from Corpus_Reader import Corpus_Reader
from collections import Counter, defaultdict
import random
import numpy as np
import sys

def random_prob():
    return random.random()*-1 + 1 # random number between 0 and 1, excluding 0, including 1


def get_uniform_parameters(corpus):
    # t(f|e) = trans_prob[e][f]
    trans_prob = defaultdict(dict)

    # q(j|i,l,m) = al_prob[(i,l,m)][j]
    al_prob = defaultdict(dict)

    # iterate over corpus to see which parameters are necessary
    for e_toks, f_toks in corpus.get_tokens_pairs():
                    l = len(e_toks)
                    m = len(f_toks)
                    for i, f_tok in enumerate(f_toks):
                        for j, e_tok in enumerate([None] + e_toks):
                            trans_prob[e_tok][f_tok] = 0
                            al_prob[(i,l,m)][j] = 0

    # reassign parameters uniformly
    for e in trans_prob:
        f_length = len(trans_prob[e])
        for f in trans_prob[e]:
            trans_prob[e][f] = 1.0/f_length

    for tpl in al_prob:
        j_length = len(al_prob[tpl])
        for j in al_prob[tpl]:
            al_prob[tpl][j] = 1.0 / j_length

    return trans_prob, al_prob


class AlignmentModel(object):
    """
    Abstract class for IBM Models that provides general methods.
    """

    def calculate_data_log_likelihood(self, corpus):
        """
        Calculates a proportional value of the log-likelihood of a corpus. Note that it is only proportional.
        """
        log_likelihood = 0
        for e_toks, f_toks in corpus:

            log_likelihood +=  self.calculate_pair_log_likelihood(f_toks=f_toks, e_toks=e_toks)

        return log_likelihood

    def calculate_perplexity(self, corpus):
        perplexity = 0
        for e_toks, f_toks in corpus:
            perplexity += self.calculate_pair_log_likelihood(f_toks=f_toks, e_toks=e_toks)
        return (perplexity / np.log(np.exp(1))) * -1


    def get_all_viterbi_alignments(self, corpus):
        all_alignments = []
        for e_toks, f_toks in corpus:
            all_alignments.append(self.get_viterbi_alignment(e_toks=e_toks, f_toks=f_toks))

        return all_alignments


class IBM1(AlignmentModel):

    def __init__(self, trans_prob=None):
        # initialize t(f|e) randomly
        # t(f|e) = trans_prob[e][f]
        if not trans_prob:
            self.trans_prob = defaultdict(lambda: defaultdict(random_prob))
        else:
            self.trans_prob = trans_prob


    def train(self, corpus, max_iterations=np.inf, convergence_ll=0.01, log_file = sys.stdout):
        """
        Trains the model using EM parameter estimation.
        """
        iteration = 0
        delta_ll = np.inf
        data_ll = - np.inf
        while(iteration < max_iterations and delta_ll > convergence_ll):
            iteration += 1
            print("Iteration ", iteration)
            # set all counts to zero
            counts_e_f = Counter()
            counts_e = Counter()
            for e_toks, f_toks in corpus.get_tokens_pairs():
                for f_tok in f_toks:
                    for e_tok in [None] + e_toks: # introduce NULL-token
                        delta = self.trans_prob[e_tok][f_tok] / np.sum([self.trans_prob[k_tok][f_tok] for k_tok in [None] + e_toks])

                        counts_e_f[(e_tok, f_tok)] += delta
                        counts_e[e_tok] += delta


            # update parameters

            for (e,f), count in counts_e_f.items():
                self.trans_prob[e][f] = count / counts_e[e]

            old_ll = data_ll
            data_ll =  self.calculate_data_log_likelihood(corpus)
            log_file.write(str(data_ll) + "\n")
            print(data_ll)
            delta_ll =  data_ll - old_ll

        if delta_ll <= convergence_ll:
            return True
        else:
            return False


    def calculate_pair_log_likelihood(self, f_toks, e_toks):
        m = len(f_toks)
        l = len(e_toks)
        sent_log_prob = -(m * np.log(l + 1))
        for i, f_tok in enumerate(f_toks):
            al_prob = 0
            for j, e_tok in enumerate([None] + e_toks):
                al_prob += self.trans_prob[e_tok][f_tok]
            sent_log_prob += np.log(al_prob)

        return sent_log_prob


    def get_viterbi_alignment(self, e_toks, f_toks):
        alignments = []
        for f_tok in f_toks:
            max_probs = []
            for e_tok in [None] + e_toks:
                max_probs.append(self.trans_prob[e_tok][f_tok])

            alignments.append(np.argmax(max_probs))

        return alignments



class IBM2(AlignmentModel):

    def __init__(self, trans_prob=None, al_prob=None):
        # initialize t(f|e) randomly
        # t(f|e) = trans_prob[e][f]
        if not trans_prob:
            self.trans_prob = defaultdict(lambda: defaultdict(random_prob))
        else:
            self.trans_prob = trans_prob

        # initialize q(j|i,l,m) randomly
        # q(j|i,l,m) = al_prob[(i,l,m)][j]
        if not al_prob:
            self.al_prob = defaultdict(lambda: defaultdict(random_prob))
        else:
            self.al_prob = al_prob

    def train(self, corpus, max_iterations=np.inf, convergence_ll=0.01, log_file=sys.stdout):
        """
        Trains the model using EM parameter estimation.
        """
        iteration = 0
        delta_ll = np.inf
        data_ll = - np.inf
        while(iteration < max_iterations and delta_ll > convergence_ll):
            iteration += 1
            print("Iteration ", iteration)
            # set all counts to zero
            counts_e_f = Counter()
            counts_e = Counter()
            counts_j_i_l_m = Counter()
            counts_i_l_m = Counter()
            for e_toks, f_toks in corpus.get_tokens_pairs():
                l = len(e_toks)
                m = len(f_toks)
                for i, f_tok in enumerate(f_toks):
                    for j, e_tok in enumerate([None] + e_toks):

                        delta = (self.al_prob[(i,l,m)][j] * self.trans_prob[e_tok][f_tok]) / np.sum([self.al_prob[(i, l, m)][k] * self.trans_prob[k_tok][f_tok] for k, k_tok in enumerate([None] + e_toks)])

                        counts_e_f[(e_tok, f_tok)] += delta
                        counts_e[e_tok] += delta
                        counts_j_i_l_m[(j, i, l, m)] += delta
                        counts_i_l_m[(i,l,m)] += delta


            # update parameters

            for (e,f), count in counts_e_f.items():
                self.trans_prob[e][f] = count / counts_e[e]

            for (j, i,l,m), count in counts_j_i_l_m.items():
                self.al_prob[(i,l,m)][j] = count / counts_i_l_m[(i,l,m)]

            # calculate likelihood
            old_ll = data_ll
            data_ll =  self.calculate_data_log_likelihood(corpus)
            log_file.write(str(data_ll) + "\n")
            print(data_ll)
            delta_ll =  data_ll - old_ll

        if delta_ll <= convergence_ll:
            return True
        else:
            return False


    def calculate_pair_log_likelihood(self, f_toks, e_toks):
        """
        Calculates the log-likelihood of a translation pair log p(f|e). Note that it is only proportional.
        """
        m = len(f_toks)
        l = len(e_toks)
        sent_log_prob = 0
        for i, f_tok in enumerate(f_toks):
            al_prob = 0
            for j, e_tok in enumerate([None] + e_toks):

                al_prob += self.trans_prob[e_tok][f_tok] * self.al_prob[(i, l, m)][j]
            sent_log_prob += np.log(al_prob)

        return sent_log_prob

    def get_viterbi_alignment(self, e_toks, f_toks):
        alignments = []
        m = len(f_toks)
        l = len(e_toks)
        for i, f_tok in enumerate(f_toks):
            max_probs = []
            for j, e_tok in enumerate([None] + e_toks):
                max_probs.append(self.trans_prob[e_tok][f_tok] * self.al_prob[(i,l,m)][j])

            alignments.append(np.argmax(max_probs))

        return alignments


class IBM1Extended(AlignmentModel):
    '''
    IBM Model 1 with Moore's enhancements:
    - Smooth rare words so they dont act as garbage collectors
    - Add probability mass to null words
    - Use heuristic initialization
    '''
    
    def __init__(self, smooth_n=1, smooth_v=46409, nr_nulls=3, llr_exponent=1, heuristic=False, trans_prob=None):
        '''
        Constructor
        '''

        self.smooth_n = smooth_n
        self.smooth_v = smooth_v
        self.nr_nulls = nr_nulls
        self.llr_exponent = llr_exponent
        self.heuristic = heuristic


        # t(f|e) = trans_prob[e][f]
        if not trans_prob:
            self.trans_prob = defaultdict(lambda: defaultdict(random_prob))
        else:
            self.trans_prob = trans_prob

    def get_llr(self, e, f):
        """
        Moore2001: "This statistic gives a measure of the likelihood that two samples are not generated
        by the same probability distribution."
        """

        llr = 0

        # e and f
        p_e_f = self.joint_counts[f][e] / self.e_counts[e]
        p_f = self.f_counts[f] / self.total_sent_count
        assert p_f > 0
        update =  self.joint_counts[f][e] * np.log(p_e_f / p_f)
        llr += update if update > 0 else 0


        # not e and f
        p_e_f = (self.f_counts[f] - self.joint_counts[f][e]) / (self.total_sent_count - self.e_counts[e])
        p_f = self.f_counts[f] / self.total_sent_count
        assert p_f > 0
        update = self.joint_counts[f][e] * np.log(p_e_f / p_f)
        llr += update if update > 0 else 0


        # e and not f
        p_e_f = (self.e_counts[e] - self.joint_counts[f][e]) / (self.total_sent_count - self.f_counts[f])
        p_f = (self.total_sent_count - self.f_counts[f]) / self.total_sent_count
        assert p_f > 0
        update =  self.joint_counts[f][e] * np.log(p_e_f / p_f)
        llr += update if update > 0 else 0



        # not e and  not f
        p_e_f = (self.total_sent_count - self.joint_counts[f][e]) / (self.total_sent_count - self.e_counts[e])
        p_f = (self.total_sent_count - self.f_counts[f]) / self.total_sent_count
        assert p_f > 0
        update = self.joint_counts[f][e] * np.log(p_e_f / p_f)
        llr += update if update > 0 else 0


        # p(t,s) > p(t) * p(s)
        p_t_s = self.joint_counts[f][e] / self.total_sent_count
        p_t = self.f_counts[f] / self.total_sent_count
        p_s = self.e_counts[e] / self.total_sent_count

        if p_t_s <= p_t * p_s:
            llr = 0

        return llr

    def train(self, corpus, max_iterations=np.inf, convergence_ll=0.01, log_file = sys.stdout):
        """
        Trains the model using EM parameter estimation.
        """

        # Heuristic initialization according to Moore
        if self.heuristic:
            # Prepare values for LLR
            # joint_counts[f][e]
            self.joint_counts = defaultdict(lambda: defaultdict(int))
            self.e_counts = defaultdict(int)
            self.f_counts = defaultdict(int)
            self.total_sent_count = 0

            self.trans_prob = defaultdict(dict)

            e_voc = set()
            f_voc = defaultdict(int)
            f_tokens = 0

            print("Getting counts")
            for e_toks, f_toks in corpus.get_tokens_pairs():
                self.total_sent_count += 1
                for e_tok in set(e_toks):
                    self.e_counts[e_tok]+= 1
                    e_voc.add(e_tok)
                    for f_tok in set(f_toks):
                        self.joint_counts[f_tok][e_tok] += 1
                        self.trans_prob[e_tok][f_tok] = 0
                        self.trans_prob[None][f_tok] = 0
                for f_tok in set(f_toks):
                    self.f_counts[f_tok] += 1
                    f_voc[f_tok] += 1
                    f_tokens += 1


            print("Getting sums")
            # initialize parameters
            max_sum = -1
            for e_tok in e_voc:
                summed = 0
                for f_tok in f_voc:
                    llr = np.power(self.get_llr(e_tok, f_tok), self.llr_exponent)
                    summed += llr

                if summed > max_sum:
                    max_sum = summed

            print("Initializing parameters")
            for e_tok in self.trans_prob:
                if e_tok is None:
                    continue
                for f_tok in self.trans_prob[e_tok]:
                    llr = np.power(self.get_llr(e_tok, f_tok), self.llr_exponent)
                    if llr >= 0.9:
                        self.trans_prob[e_tok][f_tok] = llr / max_sum
                    else:
                        self.trans_prob[e_tok][f_tok] = 0
                    # special treatment for NULL words
                    if self.nr_nulls > 0:
                        self.trans_prob[None][f_tok] = f_voc[f_tok] / f_tokens

        # for e in self.trans_prob:
        #     for f,v in self.trans_prob[e].items():
        #         print v

        iteration = 0
        delta_ll = np.inf
        data_ll = - np.inf
        while(iteration < max_iterations and delta_ll > convergence_ll):
            iteration += 1
            print("Iteration ", iteration)
            # set all counts to zero
            counts_e_f = Counter()
            counts_e = Counter()
            for e_toks, f_toks in corpus.get_tokens_pairs():
                for f_tok in f_toks:
                    for e_tok in [None]*self.nr_nulls + e_toks: # introduce NULL-token
                        delta = self.trans_prob[e_tok][f_tok] / np.sum([self.trans_prob[k_tok][f_tok] for k_tok in [None]*self.nr_nulls + e_toks])

                        counts_e_f[(e_tok, f_tok)] += delta
                        counts_e[e_tok] += delta


            # update parameters
            for (e,f), count in counts_e_f.items():
                self.trans_prob[e][f] = (count + self.smooth_n) / (counts_e[e] + self.smooth_n * self.smooth_v)

            old_ll = data_ll
            data_ll =  self.calculate_data_log_likelihood(corpus)
            log_file.write(str(data_ll) + "\n")
            print(data_ll)
            delta_ll =  data_ll - old_ll

        if delta_ll <= convergence_ll:
            return True
        else:
            return False


    def calculate_pair_log_likelihood(self, f_toks, e_toks):
        m = len(f_toks)
        l = len(e_toks)
        sent_log_prob = -(m * np.log(l + 1))
        for _, f_tok in enumerate(f_toks):
            al_prob = 0
            for _, e_tok in enumerate([None] * self.nr_nulls + e_toks):
                al_prob += self.trans_prob[e_tok][f_tok]
            sent_log_prob += np.log(al_prob)

        return sent_log_prob


    def get_viterbi_alignment(self, e_toks, f_toks):
        alignments = []
        for f_tok in f_toks:
            max_probs = []
            for e_tok in [None] + e_toks:
                max_probs.append(self.trans_prob[e_tok][f_tok])

            alignments.append(np.argmax(max_probs))

        return alignments
    
    
    

if __name__ == "__main__":

    # example usage:

    # create corpus instance
    cr = Corpus_Reader("data/corpus_1000.en", "data/corpus_1000.nl", limit=100)

    # get uniform probability distributions for parameters
    uniform_tran_probs, uniform_al_probs = get_uniform_parameters(cr)

    # create model, omit parameters for random initialization
    model = IBM1Extended(heuristic=True, llr_exponent=1.3, smooth_n=0, smooth_v=0, nr_nulls=1)

    # train model until convergence delta is small enough or max_iterations is reached
    model.train(cr, max_iterations=np.inf, convergence_ll=0.001)

    # get all translation pairs
    transl = []
    for f, v in model.trans_prob.items():
        for e, prob in v.items():
            transl.append((e,f, prob))
    # output the top 20 ones with the highest probabilities
    for e, f, prob in sorted(transl, reverse=True, key= lambda tup: tup[2])[:20]:
        print(e, f, prob)

    # get viterbi alignments
    alignments = model.get_all_viterbi_alignments(cr)

    print (alignments)


