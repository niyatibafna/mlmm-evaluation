"""
A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories
https://arxiv.org/pdf/1604.01696.pdf

'Story Cloze Test' (2018) is a commonsense reasoning framework for evaluating story
understanding, story generation, and script learning. This test requires a system
to choose the correct ending to a four-sentence story.

Homepage: https://cs.rochester.edu/nlp/rocstories/
"""
import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean


_CITATION = """
@inproceedings{sharma-etal-2018-tackling,
    title = "Tackling the Story Ending Biases in The Story Cloze Test",
    author = "Sharma, Rishi  and
      Allen, James  and
      Bakhshandeh, Omid  and
      Mostafazadeh, Nasrin",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P18-2119",
    doi = "10.18653/v1/P18-2119",
    pages = "752--757",
    abstract = "The Story Cloze Test (SCT) is a recent framework for evaluating story comprehension and script learning. There have been a variety of models tackling the SCT so far. Although the original goal behind the SCT was to require systems to perform deep language understanding and commonsense reasoning for successful narrative understanding, some recent models could perform significantly better than the initial baselines by leveraging human-authorship biases discovered in the SCT dataset. In order to shed some light on this issue, we have performed various data analysis and analyzed a variety of top performing models presented for this task. Given the statistics we have aggregated, we have designed a new crowdsourcing scheme that creates a new SCT dataset, which overcomes some of the biases. We benchmark a few models on the new dataset and show that the top-performing model on the original SCT dataset fails to keep up its performance. Our findings further signify the importance of benchmarking NLP systems on various evolving test sets.",
}
"""


class StoryCloze(Task):
    VERSION = 0
    DATASET_PATH = "story_cloze"
    DATASET_NAME = None

    NUM_FEW_SHOT = 0

    def __init__(self, data_dir: str):
        """
        StoryCloze is not publicly available. You must download the data by
        following https://cs.rochester.edu/nlp/rocstories/ and pass the folder
        path into the `data_dir` arg.
        """
        super().__init__(data_dir=data_dir)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        pass

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        '''Story Continuation and Options'''

        text = "What is a possible continuation for the following story ? \n\n"
        story = "\n".join(
            [
                doc["input_sentence_1"],
                doc["input_sentence_2"],
                doc["input_sentence_3"],
                doc["input_sentence_4"],
            ]
        ) + "\n\n"
        options = "Choose from the following options: \n" + "\n".join(
        [doc["sentence_quiz1"], doc["sentence_quiz2"]] ) + "\n\n"
        return text + story + options 
    # def doc_to_text(self, doc):
    #     '''
    #     Answer Given Options
    #     {{input_sentence_1}} {{input_sentence_2}} {{input_sentence_3}} {{input_sentence_4}}
    #     What is a possible continuation for the story given the following options ?
    #     - {{answer_choices | join("\n- ")}} ||| {{answer_choices[answer_right_ending
    #     -1]}}
    #     '''
    #     prompt = f"{doc['input_sentence_1']} {doc['input_sentence_2']} {doc['input_sentence_3']} {doc['input_sentence_4']} \n\
    #     What is a possible continuation for the story given the following options ? \n \
    #     - {doc['sentence_quiz1']} \n {doc['sentence_quiz2']}"
    #     return prompt
        


    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return " ".join(
            [
                doc["input_sentence_1"],
                doc["input_sentence_2"],
                doc["input_sentence_3"],
                doc["input_sentence_4"],
            ]
        )

    def doc_to_target(self, doc):
        clozes = [doc["sentence_quiz1"], doc["sentence_quiz2"]]
        # `- 1` because the `answer_right_ending` index is 1-based.
        return " " + clozes[doc["answer_right_ending"] - 1]

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
        clozes = [doc["sentence_quiz1"], doc["sentence_quiz2"]]
        lls = [rf.loglikelihood(ctx, " {}".format(choice))[0] for choice in clozes]
        return lls

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        gold = doc["answer_right_ending"] - 1
        acc = 1.0 if np.argmax(results) == gold else 0.0
        return {"acc": acc}

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


class StoryCloze2016(StoryCloze):
    DATASET_NAME = "2016"


class StoryCloze2018(StoryCloze):
    DATASET_NAME = "2018"
