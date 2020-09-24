import torch
import nltk

class CodraTree():

    def __init__(self, rel_type, role, left_child, right_child, raw_text, is_cnn, ann):

        if (rel_type is not None):
            self.rel_type = rel_type.replace(")","")
        else:
            self.rel_type = rel_type
        self.raw_text = raw_text
        self.role = role
        self.left_child = left_child
        self.right_child = right_child
        self.edu_text = None
        self.ann = ann

    def print_postorder(self):
        if (self.left_child is not None):
            self.left_child.print_postorder()
            self.right_child.print_postorder()
        print(self.role, self.rel_type)

    def print_postorder_eduText(self):
        if (self.left_child is not None):
            self.left_child.print_postorder_eduText()
            self.right_child.print_postorder_eduText()
        print(self.role, self.edu_text)

    def edus(self, the_edus):
        if (self.left_child is not None):
            the_edus = self.left_child.edus(the_edus)
            the_edus = self.right_child.edus(the_edus)
        if (self.edu_text is not None):
            the_edus.append(self.edu_text)
            return the_edus
        else:
            return the_edus

    def match_edus_with_sentences(self):
        sentences = self.raw_text.split(".!?")
        edus = self.edus([])
        edu_to_sentence = {}
        for edu in edus:
            clean_edu = edu.replace(".!?", "")
            for i, sentence in enumerate(sentences):
                edu_to_sentence[edu] = i
                break
        return edu_to_sentence

    def get_entities(self):
        entities = set()
        self.num_sentences = len(self.ann['sentences'])
        for sentence in self.ann['sentences']:
            dependencies = sentence['basicDependencies']
            for dependence in dependencies:
                dependent = dependence['dependent']
                token = sentence['tokens'][dependent - 1]
                pos = token['pos']
                if pos == "PRP":
                    entities.add(dependence['dependentGloss'])
                elif dependence['dep'] == 'compound':
                    entities.add(dependence['governorGloss'])
                elif pos in ["NN", "NNS", "NNP", "NNPS"]:
                    entities.add(dependence['dependentGloss'])
        return list(entities)

    def get_numSentences(self):
        return self.num_sentences

    def get_nuclei(self, entities, grid, edu_to_sentence, relation):
        if (self.edu_text is not None):
            for entity in entities:
                if entity in self.edu_text:
                    if ((relation != "Joint") and (relation != "Contrast") and (relation != "TextualOrganization") and (relation != "Same-Unit")):
                        grid[edu_to_sentence[self.edu_text]][entities.index(entity)].append(relation + "_Nucleus")
                    else:
                        grid[edu_to_sentence[self.edu_text]][entities.index(entity)].append(relation)

        else:
            if (self.left_child is not None):
                if (self.left_child.role == "Nucleus"):
                    grid = self.left_child.get_nuclei(entities, grid, edu_to_sentence, relation)
                if (self.right_child.role == "Nucleus"):
                    grid = self.right_child.get_nuclei(entities, grid, edu_to_sentence, relation)

        return grid

    def get_satellites(self, entities, grid, edu_to_sentence, relation):
        for entity in entities:
            if entity in self.edu_text:
                if ((relation != "Joint") and (relation != "Contrast") and (relation != "TextualOrganization") and (relation != "Same-Unit")):
                    grid[edu_to_sentence[self.edu_text]][entities.index(entity)].append(relation + "_Satellite")
                else:
                    grid[edu_to_sentence[self.edu_text]][entities.index(entity)].append(relation)
        return grid

    def make_grid_recurse(self, entities, grid, edu_to_sentence):
        if (self.left_child is not None):
            if (self.left_child.role == "Nucleus"):
                grid = self.left_child.get_nuclei(entities, grid, edu_to_sentence, self.right_child.rel_type)
            if (self.right_child.role == "Nucleus"):
                grid = self.right_child.get_nuclei(entities, grid, edu_to_sentence, self.left_child.rel_type)
            grid = self.left_child.make_grid_recurse(entities, grid, edu_to_sentence)
            grid = self.right_child.make_grid_recurse(entities, grid, edu_to_sentence)
        elif (self.role == "Satellite"):
            grid = self.get_satellites(entities, grid, edu_to_sentence, self.rel_type)
        return grid

    def make_grid(self):
        entities = self.get_entities()
        num_sentences = self.get_numSentences()
        grid = [[[] for x in range(len(entities))] for y in range(num_sentences)]
        return self.make_grid_recurse(entities, grid, self.match_edus_with_sentences())


def parse_codra_tree_recurse(text, raw_text, is_cnn, ann):
    """
        Builds the parse tree from the CODRA
        output
        """
    line = text[0][text[0].find("("):]
    line = line.split(" ")
    role = line[1]

    if (text[0].find("leaf") != -1):
        # Base case
        # Extract, remove the ')'
        rel_type = line[5][:len(line[5]) - 1]
        leaf = CodraTree(rel_type, role, None, None, raw_text, is_cnn, ann)
        left_idx = text[0].find("_!")
        right_idx = text[0].find("_!", left_idx + 1)
        edu_text = text[0][left_idx+2:right_idx]
        leaf.edu_text = edu_text
        return leaf, text[1:]
    else:
        # Recursive case
        # Extract, remove the ')'
        if (role != 'Root'):
            rel_type = line[6][:len(line[6]) - 1]
        else:
            rel_type = None
        left_child, text = parse_codra_tree_recurse(text[1:], raw_text, is_cnn, ann)
        right_child, text = parse_codra_tree_recurse(text, raw_text, is_cnn, ann)
        if (left_child.role == 'Nucleus'):
            left_child.rel_type = right_child.rel_type
        else:
            right_child.rel_type = left_child.rel_type
        return CodraTree(rel_type, role, left_child, right_child, raw_text, is_cnn, ann), text[1:] #skip ')'

def parse_codra_tree(filename, raw_text, is_cnn):
    """
        Returns the pointer to the root of tree representation (CodraTree)
        of CODRA's output
        """
    ann = None
    if is_cnn:
        client = CoreNLPClient(annotators=['pos','depparse'], timeout=30000, memory='16G')
        ann = client.annotate(raw_text, output_format='json')

    with open(filename, 'r') as file:
        text = file.read()
        lines = text.split("\n")

        while (lines[0].startswith("Found")):
            lines = lines[1:] # Skip "Found Unknown ..."

        return parse_codra_tree_recurse(lines, raw_text, is_cnn, ann)[0]
