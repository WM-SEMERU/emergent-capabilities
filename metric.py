from functools import partial

from bleu import _bleu
from codebleu import calc_codebleu

def _codebleu(references, predictions, lang):
    result = calc_codebleu(references, predictions, lang=lang)
    return result["codebleu"]

class Metric:
    def __init__(self, name, *, grade_single=None, grade_multi=None, baseline=0.0):
        self.name = name
        self.grade_single = grade_single
        self.grade_multi = grade_multi
        self.baseline = 0.0

    def grade(self, answer_key, answers):
        if self.grade_multi:
            return self.grade_multi(answer_key, answers)
        
        if self.grade_single:
            correct = 0
            for truth, answer in zip(answer_key, answers):
                if answer.strip() == truth.strip():
                    correct += 1
            return correct / len(answers)

        assert False, "Metric requires either grade_multi or grade_single"

ExactMatch = Metric(
    name="Accuracy% (Exact Match)",
    grade_single = lambda truth, answer: truth == answer,
)
BLEU = Metric(
    name="BLEU",
    grade_multi = _bleu,
)
CodeBLEUJava = Metric(
    name="CodeBLEU (Java)",
    grade_multi = partial(_codebleu, lang="java"),
)
CodeBLEUCSharp = Metric(
    name="CodeBLEU (C#)",
    grade_multi = partial(_codebleu, lang="c_sharp"),
)
