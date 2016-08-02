class Features(object):
    # static and dynamic features
    # static ones are stored in subcorpus file so they don't need to be extracted again
    # dynamic ones need to be accounted for (reserve a spot in feature vector) and generate weights

    def __init__(self):
        self.feature_num = 0
        self.feature_dict = dict()

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
            output += str(self.feature_dict[k]) + "\t" + "\t".join([f + " " + str(v) for (f,v) in k]) + "\n"
        return output




