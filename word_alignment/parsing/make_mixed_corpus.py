from ReorderTrees import TreeNode

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

def reorder(data, order):
    """
    Order is a list with same length as data that specifies for each position of data, which rank it has in the new order.
    :param data:
    :param order:
    :return:
    """
    new_data = [None] * len(data)
    for i, j in enumerate(order):
        new_data[j] = data[i]
    return new_data


def read_reorder_file(f):
    reorder_dict = dict()
    with open(f, "r") as infile:
        for line in infile:
            pos, left, right = map(int, line.strip().split())
            reorder_dict[pos] = (left, right)
    return reorder_dict

def make_mixed_data(f_heads, pos, order, reorder_dict={}):
    root = TreeNode(0, -1, None, None)
    actual_root = TreeNode(0, order[0], pos[0], root)
    root.right_children.append(actual_root)
    nodes = [actual_root]
    for j in xrange(1, len(f_heads)):
        p = nodes[f_heads[j]]
        n = TreeNode(j, order[j], pos[j], p)
        if order[j] < p.o:
            p.add_left_child(n)
        else:
            p.add_right_child(n)
        nodes.append(n)
    hmm_toks = []
    for j in xrange(len(nodes) - 1, -1, -1):
        n = nodes[j]
        left, right = reorder_dict.get(n.p, (0, 0))
        if left or right:
            hmm_toks += n.reorder_chain(left=left, right=right)
    actual_root = root.right_children[0]
    new_order, new_heads = zip(*actual_root.traverse_head_first())
    new_heads = map(new_order.index, new_heads)
    reordered_hmm_toks = map(new_order.index, hmm_toks)
    new_order = map(new_order.index, range(len(f_heads)))

    return new_order, new_heads, reordered_hmm_toks


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus_file", required=True)
    arg_parser.add_argument("-reorder_file", required=True)
    args = arg_parser.parse_args()
    corpus = CorpusReader(args.corpus_file)

    mixed_model = read_reorder_file(args.reorder_file)

    outfile = open(args.corpus_file + ".mixed", "w")

    for e_toks, f_toks, f_heads, pos, rel, _, order in corpus:
        new_order, f_heads, hmm_transitions = make_mixed_data(f_heads, pos, order, mixed_model)
        order = reorder(order, new_order)
        f_toks = reorder(f_toks, new_order)
        pos = reorder(pos, new_order)
        rel = reorder(rel, new_order)
        hmm_transitions = [1 if j in hmm_transitions else 0 for j in xrange(len(f_toks))]

        outfile.write(" ".join(map(str, e_toks)) + "\n")
        outfile.write(" ".join(map(str, f_toks)) + "\n")
        outfile.write(" ".join(map(str, f_heads)) + "\n")
        outfile.write(" ".join(map(str, pos)) + "\n")
        outfile.write(" ".join(map(str, rel)) + "\n")
        outfile.write(" ".join(map(str, hmm_transitions)) + "\n")
        outfile.write(" ".join(map(str, order)) + "\n\n")

    outfile.close()