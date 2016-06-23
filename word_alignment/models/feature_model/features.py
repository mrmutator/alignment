class Features(object):
    # static and dynamic features
    # static ones are stored in subcorpus file so they don't need to be extracted again
    # dynamic ones need to be accounted for (reserve a spot in feature vector) and generate weights

    def __init__(self, extract_static_func, extract_dynamic_func):
        self.feature_num = 0
        self.feature_dict = dict()
        self.extract_static = extract_static_func
        self.extract_dynamic = extract_dynamic_func

    def add(self, feat):
        if feat not in self.feature_dict:
            self.feature_dict[feat] = self.feature_num
            self.feature_num += 1
        return self.feature_dict[feat]

    def get_feat_id(self, feat):
        return self.feature_dict[feat]

    def get_voc(self):
        output = ""
        for k in sorted(self.feature_dict, key=self.feature_dict.get):
            output += str(self.feature_dict[k]) + "\t" + k + "\n"
        return output

    def load_voc(self, fname):
        tmp_dict = dict()
        with open(fname, "r") as infile:
            for line in infile:
                i, f = line.strip().split("\t")
                tmp_dict[f] = int(i)
        self.feature_dict = tmp_dict
        self.feature_num = max(tmp_dict.values())


class FeatureConditions(object):
    def __init__(self):
        self.i = 0
        self.cond_dict = dict()
        self.index_dict = dict()

    def get_id(self, feat_set):
        if feat_set not in self.cond_dict:
            self.cond_dict[feat_set] = self.i
            self.index_dict[self.i] = feat_set
            self.i += 1

        return self.cond_dict[feat_set]

    def get_feature_set(self, id):
        return self.index_dict[id]

    def get_voc(self):
        output = ""
        for k in sorted(self.cond_dict, key=self.cond_dict.get):
            output += str(self.cond_dict[k]) + "\t" + " ".join(map(str, k)) + "\n"
        return output

    def load_voc(self, fname):
        self.cond_dict = dict()
        self.index_dict = dict()
        with open(fname, "r") as infile:
            for line in infile:
                i, f = line.strip().split("\t")
                f = frozenset(map(int, f.split()))
                self.cond_dict[f] = int(i)
                self.index_dict[int(i)] = f
