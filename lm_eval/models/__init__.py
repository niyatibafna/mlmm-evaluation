from . import gpt2
from . import gpt3
from . import huggingface
from . import textsynth
from . import dummy
from . import openai

MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "hf-causal": gpt2.HFLM,
    "hf-auto": huggingface.AutoCausalLM,
    "hf-seq2seq": huggingface.AutoSeq2SeqLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "openai": openai.OpenAILM,
    "textsynth": textsynth.TextSynthLM,
    "dummy": dummy.DummyLM,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
