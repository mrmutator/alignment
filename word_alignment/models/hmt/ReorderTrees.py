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


    # This is reorder np-left only
    # def reorder(self, pos_list):
    #     reordered_toks = []
    #
    #     if self.p in pos_list:
    #         my_head = self.head
    #         if len(self.left_children) > 0:
    #             children = list(self.left_children)
    #             self.left_children = []
    #             # I have left children that become my ancestors
    #             for j in range(1, len(children)):
    #                 assert children[j].o > children[j - 1].o
    #
    #             reordered_toks.append(self.j)
    #             if self.o > my_head.o:
    #                 # I'm a right child of my head
    #                 ci = my_head.right_children.index(self)
    #                 my_head.right_children[ci] = children[0]
    #             else:
    #                 # I'm a left child
    #                 ci = my_head.left_children.index(self)
    #                 my_head.left_children[ci] = children[0]
    #
    #             children[0].head = my_head
    #
    #             for i, c in enumerate(children[1:]):
    #                 right_most = children[i].get_rightmost()
    #                 c.head = right_most
    #                 right_most.right_children.append(c)
    #                 reordered_toks.append(c.j)
    #             right_most = children[-1].get_rightmost()
    #             right_most.right_children.append(self)
    #             self.head = right_most
    #
    #     return reordered_toks

    def reorder_chain(self, left=False, right=False):
        reordered_toks = []

        if right and self.right_children:
            right_structure_ok = True
            for c in self.right_children[:-1]:
                if not c.check_strict():
                    right_structure_ok = False
            # the last one can have several
            if self.right_children[-1].left_children:
                right_structure_ok = False
            if right_structure_ok:
                right_positions = []
                for c in self.right_children[:-1]:
                    right_positions += c.traverse()
                right_positions.append((self.right_children[-1].o, self.right_children[-1]))
                for c in self.right_children[-1].left_children:
                    right_positions += c.traverse()
                _, reordered_right = zip(*sorted(right_positions, key=lambda t: t[0]))
                for i in xrange(len(reordered_right)-1):
                    reordered_right[i].right_children = [reordered_right[i+1]]
                    reordered_right[i].left_children = []
                    reordered_right[i+1].head = reordered_right[i]
                    if reordered_right[i] in self.right_children:
                        reordered_toks.append(reordered_right[i].j)


                reordered_right[-1].left_children = []
                if reordered_right[-1] in self.right_children:
                    reordered_toks.append(reordered_right[-1].j)
                self.right_children = [reordered_right[0]]
                reordered_right[0].head = self

        if left and self.left_children:
            left_structure_ok = True
            for c in self.left_children:
                if not c.check_strict():
                    left_structure_ok = False
            if left_structure_ok:
                left_positions = []
                for c in self.left_children:
                    left_positions += c.traverse()
                _, reordered_left = zip(*sorted(left_positions, key=lambda t: t[0]))
                for i in xrange(1, len(reordered_left)):
                    reordered_left[i-1].right_children = [reordered_left[i]]
                    reordered_left[i-1].left_children = []
                    reordered_left[i].head = reordered_left[i-1]
                    if reordered_left[i] in self.left_children:
                        reordered_toks.append(reordered_left[i].j)

                reordered_left[-1].right_children = [self]
                reordered_left[-1].left_children = []
                self.left_children = []

                my_head = self.head
                if self.o > my_head.o:
                    # I'm a right child of my head
                    ci = my_head.right_children.index(self)
                    head_replacement = my_head.right_children
                else:
                    # I'm a left child
                    ci = my_head.left_children.index(self)
                    head_replacement = my_head.left_children
                # head replacement is where the chain needs to be filled in

                head_replacement[ci] = reordered_left[0]
                reordered_left[0].head = my_head

                self.head = reordered_left[-1]
                reordered_toks.append(self.j)

        return reordered_toks

    def traverse_head_first(self):
        result = [(self.j, self.head.j)]
        for c in self.left_children + self.right_children:
            result += c.traverse_head_first()
        return result

    def traverse(self):
        result = [(self.o, self)]
        for c in self.left_children + self.right_children:
            result += c.traverse()
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

    def check_strict(self):
        if self.left_children:
            return False
        if len(self.right_children) > 1:
            return False
        if self.right_children:
            return self.right_children[0].check_strict()
        else:
            return True
