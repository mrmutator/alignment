class Dep_Node(object):

    def __init__(self, data=None, parent=(None, None), index=None):
        self.index = index
        if data is None:
            self.data = {}
        else:
            self.data = data
        self.parent = parent
        self.left_children = []
        self.right_children = []

    def __str__(self):
        return "Dep_Node:"+str(self.index)

    def add_parent(self, node, relation=None):
        self.parent = (node, relation)

    def add_left_child(self, node, relation=None):
        self.left_children.append((node, relation))

    def add_right_child(self, node, relation=None):
        self.right_children.append((node, relation))

    def get_child_arcs(self):
        arcs = []
        for child, rel in self.left_children:
            arcs.append((self, rel, child, "left"))
            arcs += child.get_child_arcs()
        for child, rel in self. right_children:
            arcs.append((self, rel, child, "right"))
            arcs += child.get_child_arcs()
        return arcs

    def traverse_with_heads(self):
        pairs = []
        for child, _ in self.left_children:
            pairs.append((child.index, self.index))
            pairs += child.traverse_with_heads()
        for child, _ in self.right_children:
            pairs.append((child.index, self.index))
            pairs += child.traverse_with_heads()
        return pairs

class Dependency_Tree(object):

    def __init__(self, tokens, pos_tags, root=None):
        self.root = root
        self.source_tokens = tokens
        self.source_pos_tags = pos_tags

    def get_root(self):
        return self.root

    def set_root(self, node):
        self.root = node

    def get_arcs(self):
        return self.root.get_child_arcs()

    def traverse_with_heads(self):
        pairs = []
        pairs.append((self.root.index, 0))
        pairs += self.root.traverse_with_heads()

        order, heads = zip(*pairs)
        heads_reordered = []
        for h in heads:
            if h >= 0:
                heads_reordered.append(order.index(h))
            else:
                heads_reordered.append(-1)
        tokens = map(self.source_tokens.__getitem__, order)
        return order, zip(tokens, heads_reordered)