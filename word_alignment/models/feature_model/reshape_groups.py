import gzip
import argparse
import glob
import re

class LazyFile(object):

    def __init__(self, file_name):
        self.file_name = file_name
        self.f = None

    def write(self, buffer):
        self.f = gzip.open(self.file_name, "wb")
        self.f.write(buffer)
        self.write = self.__write

    def __write(self, buffer):
        self.f.write(buffer)

    def close(self):
        if self.f:
            self.f.close()


def reshape_files(directory, num_sentences):
    files = glob.glob(directory + "/*extracted.gz.corrected.gz")
    ordered = [None] * len(files)
    for f in files:
        i = int(re.search(".*\.(\d+)\.sub_feat\.extracted\.gz\.corrected\.gz", f).group(1))
        ordered[i-1] = f
    prefix = re.search("^\./(.*?)\.\d+\.sub_feat\.", f).group(1)
    outfile_id = 1
    outfile = LazyFile(prefix + ".corpus."+str(outfile_id) + ".gz")
    sub_c = 0

    for f in ordered:
        with gzip.open(f) as infile:
            for line in infile:
                outfile.write(line)
                if line.strip() == "":
                    sub_c += 1
                if sub_c == num_sentences:
                    outfile_id += 1
                    outfile.close()
                    sub_c = 0
                    outfile = LazyFile(prefix + ".corpus." + str(outfile_id) + ".gz")

    outfile.close()





if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-dir", required=True)
    arg_parser.add_argument("-group_size", required=False, type=int, default=-1)

    args = arg_parser.parse_args()

    reshape_files(args.dir, args.group_size)



