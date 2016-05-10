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
                for j in range(1, len(children)):
                    assert children[j].o > children[j - 1].o

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
                    right_most = children[i].get_rightmost()
                    c.head = right_most
                    right_most.right_children.append(c)
                    reordered_toks.append(c.j)
                right_most = children[-1].get_rightmost()
                right_most.right_children.append(self)
                self.head = right_most

        return reordered_toks

    def traverse_head_first(self):
        result = [(self.j, self.head.j)]
        for c in self.left_children + self.right_children:
            result += c.traverse_head_first()
        return result

    def get_rightmost(self):
        if not self.right_children:
            return self
        else:
            return self.right_children[-1].get_rightmost()

    def get_structure(self):
        s = "("+str(self.o)
        for c in self.left_children + self.right_children:
            s += c.get_structure()
        s += " )"
        return s
