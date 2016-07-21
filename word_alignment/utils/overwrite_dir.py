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


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("psnt_file")
    args = arg_parser.parse_args()

    c = 0
    with open(args.psnt_file + ".corr", "w") as outfile:
        with open(args.psnt_file, "r") as infile:
         for line in infile:
             c += 1
             if c == 6:
                 J = len(line.split())
                 line = " ".join(["0"] * J) + "\n"
             elif c == 8:
                 c = 0
             outfile.write(line)

