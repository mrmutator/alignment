from collections import defaultdict
import codecs
import random
import numpy as np
import itertools

def write_params(params, out_file_name):
    outfile = open(out_file_name, "w")
    for key, value in params.iteritems():
        outfile.write(key + "\t" + str(value) + "\n")
    outfile.close()

def random_prob():
    return random.random()*-1 + 1 # random number between 0 and 1, excluding 0, including 1

def key_generator():
    length = 1
    while True:
        for tpl in itertools.product("123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", repeat=length):
            yield "".join(tpl)
        length += 1


class Vocab(object):

    def __init__(self):
        self.i = None
        self.w2index = dict()
        self.index2w = dict()
        self.generator = key_generator()
        self.__increase()

    def __increase(self):
        self.i = next(self.generator)

    def get_index(self, w):
        j = self.w2index.get(w, self.i)
        if j == self.i:
            self.w2index[w] = j
            self.index2w[j] = w
            self.__increase()
        return j

    def write_vocab(self, file_name):
        with codecs.open(file_name, "w", "utf-8") as outfile:
            for k,v in self.index2w.iteritems():
                outfile.write(str(k) + "\t" + str(v) + "\n")


class Parameters(object):

    def __init__(self, corpus, e_vocab=Vocab(), f_vocab=Vocab(),p_0=0.2):
        self.corpus = corpus
        self.cooc = set()
        self.lengths = set()
        self.e_vocab = e_vocab
        self.f_vocab = f_vocab
        self.p_0 = p_0

        self.c = 0
        self.add_corpus(corpus)

        self.params = dict()

    def add_corpus(self, corpus):
        self.c = 0
        for e_toks, f_toks in corpus:
            self.c += 1
            I = len(e_toks)
            self.lengths.add(I)
            for f_tok in f_toks:
                f = self.f_vocab.get_index(f_tok)
                self.cooc.add(("0", f))
                for i, e_tok in enumerate(e_toks):
                    self.cooc.add((self.e_vocab.get_index(e_tok), f))


    def initialize_start_randomly(self):
        for I in self.lengths:
            Z = 0
            start = np.zeros(I)
            for i in xrange(I):
                p = random_prob()
                start[i] = p
                Z += p
            Z += self.p_0 # p_0 for null word at start
            start = start / Z
            for i in xrange(I):
                self.params["b/" + str(I) + "/" + str(i)] = start[i]

    def initialize_start_uniformly(self):
        for I in self.lengths:
            Z = 0
            start = (1.0 - self.p_0) / I
            for i in xrange(I):
                self.params["b/" + str(I) + "/" + str(i)] = start


    def initialize_dist_randomly(self):
        for jmp in xrange(-max(self.lengths)+1, max(self.lengths)):
            self.params["j/" + str(jmp)] = random_prob()


    def initialize_dist_uniformly(self):
        for jmp in xrange(-max(self.lengths)+1, max(self.lengths)):
            self.params["j/"+str(jmp)] = 0.5 # number doesn't matter


    def initialize_trans_t_file(self, t_file):
        trans_dict = defaultdict(dict)
        with open(t_file, "r") as infile:
            for line in infile:
                e, f, p = line.strip().split()
                if int(e) != 0:
                    e_ = self.e_vocab.get_index(int(e))
                else:
                    e_ = "0"
                f_ = self.f_vocab.get_index(int(f))
                if (e_, f_) in self.cooc:
                    trans_dict[e_][f_] = float(p)
        for e in trans_dict:
            Z = np.sum(trans_dict[e].values())
            for f in trans_dict[e]:
                self.params["t/" + e +"/"+f] = trans_dict[e][f] / float(Z)
        del self.cooc



    def split_data_get_parameters(self, corpus, file_prefix):
        outfile_e = open(file_prefix  + ".e", "w")
        outfile_f = open(file_prefix  + ".f", "w")
        for e_toks, f_toks in corpus:
            outfile_e.write(" ".join([str(self.e_vocab.get_index(w)) for w in e_toks]) + "\n")
            outfile_f.write(" ".join([str(self.f_vocab.get_index(w)) for w in f_toks]) + "\n")

        outfile_f.close()
        outfile_e.close()

        return self.params, self.c




def prepare_data(corpus, p_0=0.2, t_file=None, num_workers=1, file_prefix="", output_vocab_file=None, random=True):
    e_vocab= Vocab()
    f_vocab = Vocab()

    parameters = Parameters(corpus, e_vocab=e_vocab, f_vocab=f_vocab, p_0=p_0)

    if random:
        parameters.initialize_dist_randomly()
        parameters.initialize_start_randomly()
    else:
        parameters.initialize_dist_uniformly()
        parameters.initialize_start_uniformly()

    parameters.initialize_trans_t_file(t_file)

    params, corpus_size = parameters.split_data_get_parameters(corpus, file_prefix=file_prefix)

    write_params(params, "params.p")

    if output_vocab_file:
        parameters.e_vocab.write_vocab(output_vocab_file + ".voc.e")
        parameters.f_vocab.write_vocab(output_vocab_file + ".voc.f")
    del parameters
    return corpus_size

if __name__ == "__main__":

    from word_alignment.utils.Corpus_Reader import GIZA_Reader
    corpus = GIZA_Reader("/media/rwechsler/Data/Reps/alignment/test/test.en_test.de.snt")
    params = prepare_data(corpus, t_file="/media/rwechsler/Data/Reps/alignment/test/t_table.txt", random=False)