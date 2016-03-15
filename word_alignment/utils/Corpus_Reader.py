__author__ = 'rwechsler'
import codecs
import re

class Corpus_Reader(object):
    """ Corpus_Reader object that allows to iterate over corresponding e-file, f-file and alignment file.
    """

    def __init__(self, e_file, f_file, al_file=None, alignment_order=('e', 'f'), limit=None, strings="int", source_dep=False):
        self.conv = int
        if strings == "unicode":
            self.conv = unicode
        elif strings == "str":
            self.conv = str

        if al_file:
            self.al_file = codecs.open(al_file, "r", "utf-8")
        else:
            self.al_file = None
        self.e_file = codecs.open(e_file, "r", "utf-8")
        self.f_file = codecs.open(f_file, "r", "utf-8")
        self.limit = limit
        if alignment_order==('e', 'f'):
            self.order = self.__convert_to_int_e_f
        elif alignment_order==('f', 'e'):
            self.order = self.__convert_to_int_f_e
        else:
            raise Exception("No valid alignment order.")

        if self.al_file:
            self.next = self.__iter_al
            self.__check_al_order()
            self.reset()
        elif source_dep:
            self.next = self.__iter_source_dep
        else:
            self.next = self.__iter_pairs


    def reset(self):
        if self.al_file:
            self.al_file.seek(0)
        self.e_file.seek(0)
        self.f_file.seek(0)

    def __check_al_order(self):
        for a,e,f in self.__iter_al():
            if len(e) != len(f):
                e_als, f_als = zip(*a)
                e_al_max = max(e_als)
                f_al_max = max(f_als)
                if max(e_al_max, f_al_max) > min(len(e)-1, len(f)-1):
                    try:
                        e[e_al_max]
                        f[f_al_max]
                    except IndexError:
                        raise Exception("Alignment order incompatible.")
                    print "passed"
                    break

    def __iter_pairs(self):
        self.reset()

        line1 = self.e_file.readline()
        line2 = self.f_file.readline()

        c = 0
        while(line1 and line2):
            c += 1
            if self.limit and c > self.limit:
                break
            yield map(self.conv, line1.split()), map(self.conv, line2.split())
            line1 = self.e_file.readline()
            line2 = self.f_file.readline()

    def __iter_source_dep(self):
        self.reset()

        line1 = self.e_file.readline()
        line2 = self.f_file.readline()

        c = 0
        while(line1 and line2):
            c += 1
            if self.limit and c > self.limit:
                break
            yield map(self.conv, line1.split()), [self.__split_tok_parent(tok)for tok in line2.split()]
            line1 = self.e_file.readline()
            line2 = self.f_file.readline()


    def __iter__(self):
        return self.next()


    def __iter_al(self):
        """Iterable that yields 3-tuple: (tokens_e, tokens_f, alignment_links). One alignment link
        is 2-tuple (position_e, position_f)."""
        self.reset()
        c = 1
        for al_line in self.al_file:
            if self.limit and c > self.limit:
                break
            c += 1
            yield map(self.order, re.findall("(\d+)-(\d+)", al_line)), map(self.conv, self.e_file.readline().strip().split(" ")), \
                  map(self.conv, self.f_file.readline().strip().split(" "))

    def __convert_to_int_e_f(self, tpl):
        return int(tpl[0]), int(tpl[1])

    def __convert_to_int_f_e(self, tpl):
        return int(tpl[1]), int(tpl[0])

    def __split_tok_parent(self, tok_):
        tok, parent = tok_.split("_")
        return self.conv(tok), int(parent)

class GIZA_Reader(object):
    """ Corpus_Reader object that allows to iterate over corresponding e-file, f-file and alignment file.
    """

    def __init__(self, giza_file, alignment_order=('e', 'f'), limit=None):
        self.giza_file = open(giza_file, "r")
        self.limit = limit
        self.next = self.__iter_pairs
        if alignment_order==('e', 'f'):
            self.e_index = 1
            self.f_index = 2
        elif alignment_order==('f', 'e'):
            self.e_index = 2
            self.f_index = 1
        else:
            raise Exception("No valid alignment order.")


    def reset(self):
        self.giza_file.seek(0)

    def __iter_pairs(self):
        self.reset()

        c = 0
        buffer = []
        for line in self.giza_file:
            buffer.append(line.strip().split())

            if len(buffer) == 3:
                yield map(int, buffer[self.e_index]), map(int, buffer[self.f_index])
                buffer = []
                c += 1
                if c == self.limit:
                    break

    def __iter__(self):
        return self.next()

if __name__ == "__main__":

    corpus = Corpus_Reader("../ALT_Lab1/data/file.en", "../ALT_Lab1/data/file.de", "../ALT_Lab1/data/file.aligned", alignment_order=("f", "e"), limit=3)
    corpus = Corpus_Reader("../ALT_Lab1/data/file.en", "../ALT_Lab1/data/file.de", alignment_order=("f", "e"), limit=3)

    for e, f in corpus:
        print e, f