from spacy.en import English
from Trees import Dependency_Tree, Dep_Node
import re

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
        tree = Dependency_Tree(tokens = [t.orth_ for t in parsed], pos_tags=[t.tag_ for t in parsed],
                               relations=[t.dep_ for t in parsed], directions=[])
        for tok in parsed:
            nodes.append(Dep_Node(index=tok.i, data={"lemma": tok.lemma_}))
        directions = []
        for tok in parsed:
            if tok.head is tok:
                tree.set_root(nodes[tok.i])
                nodes[tok.i].add_parent(None, "ROOT")
                directions.append(-1)
            else:
                nodes[tok.i].add_parent(nodes[tok.head.i], tok.dep_)
                if tok.head.i > tok.i:
                    nodes[tok.head.i].add_left_child(nodes[tok.i], tok.dep_)
                    directions.append(1)
                else:
                    nodes[tok.head.i].add_right_child(nodes[tok.i], tok.dep_)
                    directions.append(0)
        tree.directions = directions
        return tree


class StanfordParser(object):

    def __init__(self):
        pass

    def dep_parse(self, parse_output_string, strict=False):
        lines = parse_output_string.strip().split("\n")
        toks, pos = zip(*[el.split("/") for el in lines[0].split()])


        nodes = []
        tree = Dependency_Tree(tokens = toks, pos_tags=pos)
        for i, tok in enumerate(toks):
            nodes.append(Dep_Node(index=i, data={}))
        last_node = -1
        for line in lines[2:]:
            m = re.search("^(.*?)\(.*?-(\d+), .*?-(\d+)\)$", line)
            rel = m.group(1)
            head_i = int(m.group(2))-1
            child_i = int(m.group(3))-1
            if last_node + 1 != child_i:
                # missing nodes with no relation because they are punctuation
                # use heuristics to fit them into the tree
                for j in xrange(last_node+1, child_i):
                    nodes[j].add_parent(nodes[j-1], "missing")
                    nodes[j-1].add_right_child(nodes[j], "missing")

            if head_i == -1:
                tree.set_root(nodes[child_i])
                nodes[child_i].add_parent(None, "ROOT")
            else:
                if last_node == child_i:
                    if strict:
                        return None
                    continue

                nodes[child_i].add_parent(nodes[head_i], rel)
                if head_i > child_i:
                    nodes[head_i].add_left_child(nodes[child_i], rel)
                else:
                    nodes[head_i].add_right_child(nodes[child_i], rel)

            last_node = child_i

        if last_node + 1 != len(toks):
            # missing nodes with no relation because they are punctuation
            # use heuristics to fit them into the tree
            for j in xrange(last_node+1, len(toks)):
                nodes[j].add_parent(nodes[j-1], "missing")
                nodes[j-1].add_right_child(nodes[j], "missing")


        return tree