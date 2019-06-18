class Item:
    question = ""
    subject = ""
    relation = ""
    obj = ""
    questionVector = None
    candidatePool = []#candidatePool key: subject id value: list of relation id

    def __init__(self,qid, question, sub, relation,subject_text, obj,anonymous_question,cand_rel):
        self.qid = qid
        self.question = question
        self.subject = sub
        self.relation = relation
        self.object = obj
        self.subject_text = subject_text
        self.anonymous_question = anonymous_question
        self.cand_rel = cand_rel
