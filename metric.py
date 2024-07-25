from functools import partial

from bleu import _bleu
from codebleu import calc_codebleu
import logging

def _codebleu(references, predictions, lang):
    result = calc_codebleu(references, predictions, lang=lang)
    return result["codebleu"]

# default _bleu: smooth=True, lower=False

def b_moses(ref, trans):
    return _bleu(ref, trans, smooth=False, lower=False)

def b_norm(ref, trans):
    return _bleu(ref, trans, smooth=True, lower=True)

class Metric:
    Directory = {}
    def __init__(self, name, simplename, shortname, latex_name, *, grade_single=None, grade_multi=None, baseline=0.0):
        self.name = name
        self.simplename = simplename
        self.shortname = shortname
        self.latex_name = latex_name
        self._grade_single = grade_single
        self._grade_multi = grade_multi
        self.baseline = 0.0
        Metric.Directory[self.shortname] = self

    def grade(self, answer_key, answers):
        if self._grade_multi:
            return self._grade_multi(answer_key, answers)
        
        if self._grade_single:
            correct = 0
            for truth, answer in zip(answer_key, answers):
                if self._grade_single(answer.strip(), truth.strip()):
                    correct += 1
            return correct / len(answers)

        assert False, "Metric requires either grade_multi or grade_single"

    def grade_single(self, truth, answer, silent=False):
        logger = logging.getLogger()
        logger_disabled = logger.disabled
        if silent and not logger_disabled:
            logger.disabled = True
        
        if self._grade_multi:
            value = self._grade_multi([ truth ], [ answer ])
        elif self._grade_single:
            value = self._grade_single(truth, answer)
        else:
            assert False, "Metric requires either grade_multi or grade_single"

        if silent and not logger_disabled:
            logger.disabled = False

        return value


    @staticmethod
    def from_shortname(shortname):
        return Metric.Directory[shortname]

ExactMatch = Metric(
    #name="Accuracy% (Exact Match)",
    #simplename="Accuracy%",
    name="Exact Match",
    simplename="Exact Match",
    shortname="em",
    latex_name="EM",
    grade_single = lambda truth, answer: truth == answer,
)
BLEU = Metric(
    name="BLEU",
    simplename="BLEU",
    shortname="bleu",
    latex_name="BLEU",
    grade_multi = _bleu,
)
CodeBLEUJava = Metric(
    name="CodeBLEU (Java)",
    simplename="CodeBLEU",
    shortname="codebleu-java",
    latex_name="CodeBLEU (Java)",
    grade_multi = partial(_codebleu, lang="java"),
)
CodeBLEUCSharp = Metric(
    name="CodeBLEU (C#)",
    simplename="CodeBLEU",
    shortname="codebleu-cs",
    latex_name="CodeBLEU (C$^\sharp$)",
    grade_multi = partial(_codebleu, lang="c_sharp"),
)
BMoses = Metric(
    name="B-Moses",
    simplename="B-Moses",
    shortname="codebleu-bmoses",
    latex_name="B-Moses",
    grade_multi = b_moses,
)
BNorm = Metric(
    name="B-Norm",
    simplename="B-Norm",
    shortname="codebleu-bnorm",
    latex_name="B-Norm",
    grade_multi = b_norm,
)
