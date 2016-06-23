class CorpusReader(object):
    def __init__(self, corpus_file, limit=None):
        self.corpus_file = open(corpus_file, "r")
        self.limit = limit
        self.next = self.__iter_sent

    def reset(self):
        self.corpus_file.seek(0)

    def __iter_sent(self):
        self.reset()
        c = 0
        buffer = []
        b = 0
        for line in self.corpus_file:
            if b != -1:
                buffer.append(map(int, line.strip().split()))
            b += 1
            if b == 7:
                yield buffer
                c += 1
                if c == self.limit:
                    break
                b = -1
                buffer = []


    def __iter__(self):
        return self.next()

    def get_length(self):
        c = 0
        for _ in self:
            c += 1
        return c


class SubcorpusReader(object):
    def __init__(self, corpus_file, limit=None):
        self.corpus_file = open(corpus_file, "r")
        self.limit = limit
        self.next = self.__iter_sent

    def reset(self):
        self.corpus_file.seek(0)

    def __iter_sent(self):
        self.reset()
        c = 0
        while True:
            print c+1
            e_toks = map(int, self.corpus_file.readline().strip().split())
            f_toks = map(int, self.corpus_file.readline().strip().split())
            if not f_toks:
                break
            f_heads = map(int, self.corpus_file.readline().strip().split())
            feature_sets = []
            els = map(int, self.corpus_file.readline().strip().split())
            static_cond_id = els[0]
            feature_sets.append([(static_cond_id, els[1:])])
            I =len(e_toks)
            for _ in xrange(1, len(f_toks)):
                j_sets = []
                for _ in xrange(I):
                    els = map(int, self.corpus_file.readline().strip().split())
                    j_ip_static_cond_id = els[0]
                    j_sets.append((j_ip_static_cond_id, els[1:]))
                feature_sets.append(j_sets)
            self.corpus_file.readline()
            yield (e_toks, f_toks, f_heads, feature_sets)
            c += 1
            if c == self.limit:
                break


    def __iter__(self):
        return self.next()

    def get_length(self):
        c = 0
        for _ in self:
            c += 1
        return c

class Corpus_Buffer(object):
    def __init__(self, corpus, buffer_size=20):
        self.buffer_size = buffer_size
        self.corpus = corpus

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


if __name__ == "__main__":
    pass
