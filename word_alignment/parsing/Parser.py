from spacy.en import English
from Trees import Dependency_Tree, Dep_Node

class Spacy_Parser(object):

    def __init__(self):
        self.parser = English()

    def dep_parse(self, tokens):
        if isinstance(tokens, list):
            parsed = self.parser(" ".join(tokens))
            if not len(parsed) == len(tokens):
                return None
        else:
            parsed = self.parser(tokens)

        if len(list(parsed.sents)) > 1:
            return None

        nodes = []
        tree = Dependency_Tree(tokens = [t.orth_ for t in parsed], pos_tags=[t.tag_ for t in parsed])
        for tok in parsed:
            nodes.append(Dep_Node(index=tok.i, data={"lemma": tok.lemma_}))
        for tok in parsed:
            if tok.head is tok:
                tree.set_root(nodes[tok.i])
                nodes[tok.i].add_parent(None, "ROOT")
            else:
                nodes[tok.i].add_parent(nodes[tok.head.i], tok.dep_)
                if tok.head.i > tok.i:
                    nodes[tok.head.i].add_left_child(nodes[tok.i], tok.dep_)
                else:
                    nodes[tok.head.i].add_right_child(nodes[tok.i], tok.dep_)

        return tree