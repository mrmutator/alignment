import argparse
from CorpusReader import CorpusReader, Corpus_Buffer
import features
import gzip



def extract_features(corpus_buffer, out_file_name):
    feature_voc = features.Features()
    vector_ids = features.FeatureConditions()
    outfile = gzip.open(out_file_name + ".extracted.gz", "w")
    for buff in corpus_buffer:
        for e_toks, f_toks, f_heads, pos, rel, dir, order in buff:
            J = len(f_toks)
            I = len(e_toks)

            outfile.write(" ".join(map(str, e_toks)) + "\n")
            outfile.write(" ".join(map(str, f_toks)) + "\n")
            outfile.write(" ".join(map(str, f_heads)) + "\n")
            # For debugging:
            # outfile.write(" ".join(["-1" for _ in e_toks]) + "\n")
            # outfile.write(" ".join(["-2" for _ in f_toks]) + "\n")
            # outfile.write(" ".join(["-3" for _ in f_heads]) + "\n")

            I_ = 1
            for j in xrange(J):
                for i_p in xrange(I_):
                    # conditions
                    conditions = ["empty"]

                    # STATIC FEATURE EXTRACTION HERE
                    # don't use spaces in feature name
                    # fname must not be a bare number
                    fname = "pos=" + str(pos[j])
                    conditions.append(fname)
                    fname = "rel=" + str(rel[j])
                    conditions.append(fname)


                    con_id = vector_ids.get_id(tuple(conditions))
                    outfile.write(str(con_id) + " ")

                    for i in xrange(I):
                        dynamic_features = []
                        dynamic_cond = []

                        # DYNAMIC FEATURE EXTRACTION HERE
                        fname = "jmp=" + str(i - i_p)
                        dynamic_features.append(fname)

                        for fname in dynamic_features:
                            f_id = feature_voc.add(fname)
                            dynamic_cond.append(f_id)
                        cond_id = vector_ids.get_id(frozenset(dynamic_cond))
                        outfile.write(str(cond_id) + " ")
                    outfile.write("\n")
                    I_ = I

            outfile.write("\n")

    outfile.close()

    with open(out_file_name + ".fvoc", "w") as outfile:
        outfile.write(feature_voc.get_voc())

    with open(out_file_name + ".convoc", "w") as outfile:
        outfile.write(vector_ids.get_voc())


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    args = arg_parser.parse_args()
    corpus = CorpusReader(args.corpus)
    corpus_buffer = Corpus_Buffer(corpus, buffer_size=200)

    extract_features(corpus_buffer, args.corpus)