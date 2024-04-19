"""
XNLI: Evaluating Cross-lingual Sentence Representations
https://arxiv.org/abs/1809.05053

Based on the implementation of @yongzx (see https://github.com/EleutherAI/lm-evaluation-harness/pull/258)

Prompt format (same as XGLM and mGPT):

sentence1 + ", right? " + mask = (Yes|Also|No) + ", " + sentence2

Predicition is the full sequence with the highest likelihood.

Language specific prompts are translated word-by-word with Google Translate
and may differ from the ones used by mGPT and XGLM (they do not provide their prompts).

Homepage: https://github.com/facebookresearch/XNLI
"""
import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
from lm_eval import utils

_CITATIONS = """
@InProceedings{conneau2018xnli,
  author = "Conneau, Alexis
        and Rinott, Ruty
        and Lample, Guillaume
        and Williams, Adina
        and Bowman, Samuel R.
        and Schwenk, Holger
        and Stoyanov, Veselin",
  title = "XNLI: Evaluating Cross-lingual Sentence Representations",
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods
               in Natural Language Processing",
  year = "2018",
  publisher = "Association for Computational Linguistics",
  location = "Brussels, Belgium",
}
"""


class XNLIBase(Task):
    VERSION = 0
    DATASET_PATH = "xnli"
    DATASET_NAME = None

    QUESTION_WORD = None  # 'right'
    ENTAILMENT_LABEL = None  # 'Yes'
    NEUTRAL_LABEL = None  # 'Also'
    CONTRADICTION_LABEL = None  # 'No'

    NUM_FEW_SHOT = 10

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        # Example:
        # The girl that can help me is all the way across town, right? Yes, The girl I need help from lives a ways away.
        # [MASK] is replaced with ENTAILMENT_LABEL, NEUTRAL_LABEL, or CONTRADICTION_LABEL
        return (
            doc["premise"]
            + ", "
            + self.QUESTION_WORD
            + "? [MASK], "
            + doc["hypothesis"]
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return (
            " "
            + [self.ENTAILMENT_LABEL, self.NEUTRAL_LABEL, self.CONTRADICTION_LABEL][
                doc["label"]
            ]
        )

    def doc_to_fewshot_prompt(self, doc):

        prompt = self.doc_to_text(doc)
        return prompt.replace("[MASK]", self.doc_to_target(doc)[1:])

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        ll_true = rf.loglikelihood_rolling(ctx.replace("[MASK]", self.ENTAILMENT_LABEL))
        ll_neither = rf.loglikelihood_rolling(ctx.replace("[MASK]", self.NEUTRAL_LABEL))
        ll_false = rf.loglikelihood_rolling(
            ctx.replace("[MASK]", self.CONTRADICTION_LABEL)
        )

        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        gold = doc["label"]
        pred = np.argmax(results)
        return {"acc": pred == gold}

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {"acc": mean}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {"acc": True}

    @utils.positional_deprecated
    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            )

        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs()
                        if self.has_validation_docs()
                        else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = (
                "\n\n".join(
                    [
                        # self.doc_to_text(doc) + self.doc_to_target(doc)
                        self.doc_to_fewshot_prompt(doc)
                        for doc in fewshotex
                    ]
                )
                + "\n\n"
            )

        example = self.doc_to_text(doc)
        return description + labeled_examples + example


class XNLI_en(XNLIBase):  # English
    DATASET_NAME = "en"

    QUESTION_WORD = "right"
    ENTAILMENT_LABEL = "Yes"
    NEUTRAL_LABEL = "Also"
    CONTRADICTION_LABEL = "No"


class XNLI_de(XNLIBase):  # German
    DATASET_NAME = "de"

    QUESTION_WORD = "richtig"
    ENTAILMENT_LABEL = "Ja"
    NEUTRAL_LABEL = "Auch"
    CONTRADICTION_LABEL = "Nein"


class XNLI_ar(XNLIBase):  # Arabic
    DATASET_NAME = "ar"

    QUESTION_WORD = "صحيح"
    ENTAILMENT_LABEL = "نعم"
    NEUTRAL_LABEL = "لذا"
    CONTRADICTION_LABEL = "رقم"


class XNLI_bg(XNLIBase):  # Bulgarian
    DATASET_NAME = "bg"

    QUESTION_WORD = "правилно"
    ENTAILMENT_LABEL = "да"
    NEUTRAL_LABEL = "така"
    CONTRADICTION_LABEL = "не"


class XNLI_el(XNLIBase):  # Greek
    DATASET_NAME = "el"

    QUESTION_WORD = "σωστός"
    ENTAILMENT_LABEL = "Ναί"
    NEUTRAL_LABEL = "Έτσι"
    CONTRADICTION_LABEL = "όχι"


class XNLI_es(XNLIBase):  # Spanish
    DATASET_NAME = "es"

    QUESTION_WORD = "correcto"
    ENTAILMENT_LABEL = "Sí"
    NEUTRAL_LABEL = "Asi que"
    CONTRADICTION_LABEL = "No"


class XNLI_fr(XNLIBase):  # French
    DATASET_NAME = "fr"

    QUESTION_WORD = "correct"
    ENTAILMENT_LABEL = "Oui"
    NEUTRAL_LABEL = "Aussi"
    CONTRADICTION_LABEL = "Non"


class XNLI_hi(XNLIBase):  # Hindi
    DATASET_NAME = "hi"

    QUESTION_WORD = "सही"
    ENTAILMENT_LABEL = "हाँ"
    NEUTRAL_LABEL = "इसलिए"
    CONTRADICTION_LABEL = "नहीं"


class XNLI_ru(XNLIBase):  # Russian
    DATASET_NAME = "ru"

    QUESTION_WORD = "правильно"
    ENTAILMENT_LABEL = "Да"
    NEUTRAL_LABEL = "Так"
    CONTRADICTION_LABEL = "Нет"


class XNLI_sw(XNLIBase):  # Swahili
    DATASET_NAME = "sw"

    QUESTION_WORD = "sahihi"
    ENTAILMENT_LABEL = "Ndiyo"
    NEUTRAL_LABEL = "Hivyo"
    CONTRADICTION_LABEL = "Hapana"


class XNLI_th(XNLIBase):  # Thai
    DATASET_NAME = "th"

    QUESTION_WORD = "ถูกต้อง"
    ENTAILMENT_LABEL = "ใช่"
    NEUTRAL_LABEL = "ดังนั้น"
    CONTRADICTION_LABEL = "ไม่"


class XNLI_tr(XNLIBase):  # Turkish
    DATASET_NAME = "tr"

    QUESTION_WORD = "doğru"
    ENTAILMENT_LABEL = "Evet"
    NEUTRAL_LABEL = "Böylece"
    CONTRADICTION_LABEL = "Hayır"


class XNLI_ur(XNLIBase):  # Urdu
    DATASET_NAME = "ur"

    QUESTION_WORD = "صحیح"
    ENTAILMENT_LABEL = "جی ہاں"
    NEUTRAL_LABEL = "اس لئے"
    CONTRADICTION_LABEL = "نہیں"


class XNLI_vi(XNLIBase):  # Vietnamese
    DATASET_NAME = "vi"

    QUESTION_WORD = "đúng"
    ENTAILMENT_LABEL = "Vâng"
    NEUTRAL_LABEL = "Vì vậy"
    CONTRADICTION_LABEL = "Không"


class XNLI_zh(XNLIBase):  # Chinese
    DATASET_NAME = "zh"

    QUESTION_WORD = "正确"
    ENTAILMENT_LABEL = "是的"
    NEUTRAL_LABEL = "所以"
    CONTRADICTION_LABEL = "不是的"


LANGS = [
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
]

LANG_CLASSES = [
    XNLI_ar,
    XNLI_bg,
    XNLI_de,
    XNLI_el,
    XNLI_en,
    XNLI_es,
    XNLI_fr,
    XNLI_hi,
    XNLI_ru,
    XNLI_sw,
    XNLI_th,
    XNLI_tr,
    XNLI_ur,
    XNLI_vi,
    XNLI_zh,
]


def construct_tasks():
    tasks = {}
    for lang, lang_class in zip(LANGS, LANG_CLASSES):
        tasks[f"xnli_{lang}"] = lang_class
    return tasks
