# Reminder: don't confuse j with actual order[j]


def extract_static_dist_features(e_toks, f_toks, f_heads, pos, rel, order, j, i_p):
    """
    Extracts static features for a given position j.
    Returns a list that contains the feature names which
    have a True binary value.
    """
    feature_statements = []
    # tree_level = [0]
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


def extract_dynamic_dist_features(e_toks, f_toks, f_heads, pos, rel, order, j, i_p, i):
    """
    Extracts dynamic features for a given position j.
    THIS FUNCTION MUST ONLY RETURN FEATURES THAT DEPEND ON i.
    Returns a list that contains the feature names which
    have a True binary value.
    """
    feature_statements = []
    # tree_level = [0]
    # par = f_heads[j]
    # tree_level.append(tree_level[par] + 1)
    # orig_tok_pos = order[j]
    # orig_head_pos = order[par]
    # parent_distance = abs(orig_head_pos - orig_tok_pos)
    # pos[par], rel[par], dir[par], pos[j], rel[j], dir[j], parent_distance, tree_level[j]
    fname = "jump=" + str(i - i_p)
    feature_statements.append(fname)

    return feature_statements
