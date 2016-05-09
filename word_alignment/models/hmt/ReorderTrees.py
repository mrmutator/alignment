class TreeNode(object):
    def __init__(self, j, o, p, head):
        self.j = j
        self.o = o
        self.p = p
        self.head = head
        self.left_children = []
        self.right_children = []

    def add_left_child(self, c):
        self.left_children.append(c)

    def add_right_child(self, c):
        self.right_children.append(c)

    def reorder(self, pos_list):
        reordered_toks = []

        if self.p in pos_list:
            my_head = self.head
            if len(self.left_children) > 0:
                children = list(self.left_children)
                self.left_children = []
                # I have left children that become my ancestors
                for j in range(1, len(self.left_children)):
                    assert self.left_children[j].o > self.left_children[j - 1].o

                reordered_toks.append(self.j)
                if self.o > my_head.o:
                    # I'm a right child of my head
                    ci = my_head.right_children.index(self)
                    my_head.right_children[ci] = children[0]
                else:
                    # I'm a left child
                    ci = my_head.left_children.index(self)
                    my_head.left_children[ci] = children[0]

                children[0].head = my_head

                for i, c in enumerate(children[1:]):
                    c.head = children[i]
                    children[i].right_children.append(c)
                    reordered_toks.append(c.j)
                children[-1].right_children.append(self)
                self.head = children[-1]

        return reordered_toks

    def traverse_head_first(self):
        result = [(self.j, self.head.j)]
        for c in self.left_children + self.right_children:
            result += c.traverse_head_first()
        return result


if __name__ == "__main__":

    def make_mixed_data(f_heads, pos, order, reorder_tags=[]):
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
            hmm_toks += nodes[j].reorder(reorder_tags)
        new_order, new_heads = zip(*actual_root.traverse_head_first())
        new_heads = map(new_order.index, new_heads)
        reordered_hmm_toks = map(new_order.index, hmm_toks)
        new_order = map(new_order.index, range(len(f_heads)))

        return new_order, new_heads, reordered_hmm_toks


    def reorder(data, order):
        new_data = [None] * len(data)
        for i, j in enumerate(order):
            new_data[j] = data[i]
        return new_data


    f_toks = ["lead", "program", "a", "greater", "building", "might", "to"]
    dir = [-1, 0, 0, 0, 0, 0, 1]
    rel = ["r", "sbj", "det", "adj", "nn", "mod", "nprep"]
    f_heads = [0, 0, 1, 1, 1, 0, 0]
    pos = [0, 1, 0, 0, 0, 0, 0]
    order = [5, 3, 0, 1, 2, 4, 6]
    reorder_tags = [1]

    new_order, f_heads, hmm_transitions = make_mixed_data(f_heads, pos, order, reorder_tags)
    order = reorder(order, new_order)
    f_toks = reorder(f_toks, new_order)
    pos = reorder(pos, new_order)
    dir = reorder(dir, new_order)
    rel = reorder(rel, new_order)

    print new_order
    print f_toks
    print order
    print pos
    print dir
    print rel
    print hmm_transitions
