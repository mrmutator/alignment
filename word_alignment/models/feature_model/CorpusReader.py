class CorpusReader(object):
    def __init__(self, corpus_file):
        self.corpus_file = open(corpus_file, "r")
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
    def __init__(self, corpus_file):
        self.corpus_file = open(corpus_file, "r")
        self.next = self.__iter_sent

    def reset(self):
        self.corpus_file.seek(0)

    def __iter_sent(self):
        self.reset()
        c = 0
        while True:
            e_toks = map(int, self.corpus_file.readline().split())
            f_toks = map(int, self.corpus_file.readline().split())
            if not f_toks:
                break
            f_heads = map(int, self.corpus_file.readline().split())
            J = len(f_toks)
            feature_sets = [None] * J
            feature_sets[0] = self.corpus_file.readline().strip()
            I =len(e_toks)
            for j in xrange(1, J):
                j_sets = [None] * I
                for i in xrange(I):
                    j_sets[i] = self.corpus_file.readline().strip()
                feature_sets[j] = j_sets
            self.corpus_file.readline()
            yield (e_toks, f_toks, f_heads, feature_sets)
            c += 1


    def __iter__(self):
        return self.next()

    def get_length(self):
        c = 0
        for _ in self:
            c += 1
        return c


if __name__ == "__main__":
    pass
