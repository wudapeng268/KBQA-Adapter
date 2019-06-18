class Item:
    def __init__(self,qid, question, sub, relation, obj, gold_type, subject_text, anonymous_question):
        self.qid = qid
        self.question = question
        self.subject = sub
        self.relation = relation
        self.object = obj
        self.gold_type = gold_type
        self.subject_text = subject_text
        self.anonymous_question = anonymous_question
        self.num_text_token = len(question.split(" "))

    def add_candidate(self, sub, rels, types=None):
        if not hasattr(self, 'cand_sub'):
            self.cand_sub = []
        if not hasattr(self, 'cand_rel'):
            self.cand_rel = []
        if not hasattr(self, 'sub_rels'):
            self.sub_rels = []
        self.cand_sub.append(sub)
        self.sub_rels.append(rels)
        self.cand_rel.extend(rels)
        if types:
            if not hasattr(self, 'sub_types'):
                self.sub_types = []
            self.sub_types.append(types)

    def remove_duplicate(self):
        if hasattr(self,'cand_rel'):
            self.cand_rel = list(set(self.cand_rel))

