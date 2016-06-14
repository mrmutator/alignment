# Reminder: don't confuse j with actual order[j]

def extract_static_dist_features(e_toks, f_toks, f_heads, pos, rel, dir, order, j):
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

def extract_dynamic_dist_features(e_toks, f_toks, f_heads, pos, rel, dir, order, j, i_p):
    """
    Extracts dynamic features for a given j and i_p (alignment point of parent of j).
    Returns a list that contains the feature names which
    have a True binary value.
    Only use features here that cannot be extracted beforehand as a preprocessing step.
    """
    feature_statements= []
    if i_p == 0:
        feature_statements.append("aligned_to_first")
    if i_p == len(e_toks)-1:
        feature_statements.append("aligned_to_last")
    if i_p >= len(e_toks):
        feature_statements.append("aligned_to_null")
    return feature_statements


def extract_static_start_features(e_toks, f_toks, f_heads, pos, rel, dir, order):
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
