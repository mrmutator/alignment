class CorpusReader(object):
    def __init__(self, corpus_file):
        self.corpus_file = open(corpus_file, "r")
        self.__generator = self.iter_sent()
        self.next = self.__generator.next
        self.__iter__ = self.__generator

    def reset(self):
        self.corpus_file.seek(0)

    def iter_sent(self):
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

    def get_length(self):
        c = 0
        for _ in self:
            c += 1
        return c


class AnnotatedCorpusReader(object):
    def __init__(self, corpus_file):
        self.corpus_file = open(corpus_file, "r")
        self.__generator = self.__iter_sent()
        self.next = self.__generator.next

    def reset(self):
        self.corpus_file.seek(0)

    def __iter_sent(self):
        self.reset()
        c = 0
        buffer = []
        b = 0
        for line in self.corpus_file:
            if b != -1:
                if b < 9:
                    buffer.append(map(int, line.strip().split()))
                else:
                    buffer.append(line.strip().split())
            b += 1
            if b == 11:
                yield buffer
                c += 1
                b = -1
                buffer = []

    def __iter__(self):
        return self.__generator

    def get_length(self):
        c = 0
        for _ in self:
            c += 1
        return c


class SubcorpusReader(object):
    def __init__(self, corpus_file):
        self.corpus_file = open(corpus_file, "r")
        self.__generator = self.__iter_sent()
        self.next = self.__generator.next

    def reset(self):
        self.corpus_file.seek(0)

    def __iter_sent(self):
        self.reset()
        c = 0
        while True:
            I = self.corpus_file.readline().strip()
            if not I:
                break
            I = int(I)
            f_heads = map(int, self.corpus_file.readline().split())
            gold_aligned = map(int, self.corpus_file.readline().split())
            J = len(f_heads)
            feature_sets = [None] * J
            feature_sets[0] = self.corpus_file.readline().split()
            for j in xrange(1, J):
                j_sets = [None] * (I+1)
                for i in xrange(I+1):
                    j_sets[i] = self.corpus_file.readline().split()
                feature_sets[j] = j_sets
            self.corpus_file.readline()
            yield (I, f_heads, gold_aligned, feature_sets)
            c += 1


    def __iter__(self):
        return self.__generator

    def get_length(self):
        c = 0
        for _ in self:
            c += 1
        return c



if __name__ == "__main__":
    pass
