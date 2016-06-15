# Reminder: don't confuse j with actual order[j]

def extract_dist_features(e_toks, f_toks, f_heads, pos, rel, order, j, i_p):
    """
    Extracts static features for a given position j.
    Returns a list that contains the feature names which
    have a True binary value.
    """
    feature_statements = []
    #tree_level = [0]
    # par = f_heads[j]
    # tree_level.append(tree_level[par] + 1)
    # orig_tok_pos = order[j]
    # orig_head_pos = order[par]
    # parent_distance = abs(orig_head_pos - orig_tok_pos)
    # pos[par], rel[par], dir[par], pos[j], rel[j], dir[j], parent_distance, tree_level[j]
    fname = "pos=" + str(pos[j])
    feature_statements.append(fname)
    fname = "rel=" + str(rel[j])
    feature_statements.append(fname)

    return feature_statements

def extract_start_features(e_toks, f_toks, f_heads, pos, rel, order):
    """
    Extracts static features for start position
    Returns a list that contains the feature names which
    have a True binary value.
    """
    j = 0
    feature_statements = []
    #tree_level = [0]
    # par = f_heads[j]
    # tree_level.append(tree_level[par] + 1)
    # orig_tok_pos = order[j]
    # orig_head_pos = order[par]
    # parent_distance = abs(orig_head_pos - orig_tok_pos)
    # pos[par], rel[par], dir[par], pos[j], rel[j], dir[j], parent_distance, tree_level[j]
    fname = "pos=" + str(pos[j])
    feature_statements.append(fname)
    fname = "rel=" + str(rel[j])
    feature_statements.append(fname)

    return feature_statements

class Features(object):
    # static and dynamic features
    # static ones are stored in subcorpus file so they don't need to be extracted again
    # dynamic ones need to be accounted for (reserve a spot in feature vector) and generate weights

    def __init__(self, extract_func=lambda:[]):
        self.feature_num = 0
        self.feature_dict = dict()
        self.extract = extract_func

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


