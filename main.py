import argparse
import glob
import json
import logging
import fnmatch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from collections import defaultdict

if os.path.exists('/mnt/localssd/'):
    os.environ['TRANSFORMERS_CACHE'] = '/mnt/localssd/cache'

from lm_eval import tasks, evaluator

import sys
sys.path.append('/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/')
sys.path.append('/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/noisers/')
from noisers.main import parse_noise_params, get_noisers
from noisers.utils.misc import get_lang_name


logging.getLogger("openai").setLevel(logging.WARNING)


def _is_json_task(task_name):
    return task_name == "json" or task_name.startswith("json=")


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0 and not _is_json_task(
                    value
            ):
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='hf-auto')
    parser.add_argument("--model_alias", type=str, default='okapi-rlhf')
    parser.add_argument("--task_alias", type=str, default='open_llm')
    parser.add_argument("--model_args", type=str, required=True)
    parser.add_argument("--tasks", default='arc_vi,mmlu_vi,hellaswag_vi', required=True)
    parser.add_argument("--provide_description", action="store_true")
    # parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--dataset_outfile", type=str, default=None)

    # Noise parameters
    parser.add_argument("--all_noise_params_str", type=str, default="")

    return parser.parse_args()


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        if _is_json_task(pattern):
            task_names.add(pattern)

        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return sorted(list(task_names))

def build_description_dict(task_names):
    '''
    Build a description dictionary for the tasks in task_names, containing custom descriptions for each task.
    '''
    description_dict = dict()
    for task_name in task_names:
        if "flores" in task_name:
            # Get the language code from the task name
            src_lang_code = task_name.split("-")[1]
            tgt_lang_code = task_name.split("-")[2]
            src_lang_name = get_lang_name(src_lang_code)
            tgt_lang_name = get_lang_name(tgt_lang_code)
            description_dict[task_name] = f"Translate from {src_lang_name} to {tgt_lang_name}:\n"
        elif "story_cloze" in task_name:
            # description_dict[task_name] = "Choose the correct ending to the story: "
            description_dict[task_name] = ""
        elif "nli" in task_name:
            description_dict[task_name] = ""
        ### TODO: Add custom descriptions for other tasks
        elif "arc" in task_name:
            description_dict[task_name] = "Choose the correct answer:"
        elif "hellaswag" in task_name:
            description_dict[task_name] = "Choose the correct continuation:"
        elif "mmlu" in task_name:
            description_dict[task_name] = "Choose the correct answer:"
        else:
            description_dict[task_name] = ""
        print(f"TASK DESCRIPTION ON FOLLOWING LINE:")
        print(description_dict[task_name])

    return description_dict


def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
    task_names = args.tasks.split(',')
    task_names = pattern_match(task_names, tasks.ALL_TASKS)

    ### TODO: Format output file and results according to required organization
    output_filename = f'{args.task_alias}-{args.model_alias}.json'
    output_file = os.path.join('logs', output_filename)
    existing_output_files = glob.glob('logs/*.json') + glob.glob('logs/*/*.json')
    existing_filenames = [os.path.basename(x) for x in existing_output_files]

    # if output_filename in existing_filenames:
        # i = existing_filenames.index(output_filename)
        # print(f"Skipping {args.task_alias}. Log file exists at {existing_output_files[i]}")
        # return

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)
    else:
        description_dict = build_description_dict(task_names)

    # Parse noise parameters e.g. phonological:theta_1-0.5;syntax:theta_2-0.5
    all_noise_params = parse_noise_params(args.all_noise_params_str)
    print(f"Noise Parameters: {all_noise_params}")
    noiser_classes = get_noisers(all_noise_params)

    results = evaluator.open_llm_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
        noiser_classes=noiser_classes,
        dataset_outfile=args.dataset_outfile,
    )

    ### TODO: Format output file and results according to required organization

    # dumped = json.dumps(results, indent=2)
    # with open(output_file, 'w') as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        if os.path.exists(args.output_path):
            with open(args.output_path, "r") as f:
                file = json.load(f)
                file[args.all_noise_params_str] = results
        else:
            file = {args.all_noise_params_str: results}
        with open(args.output_path, "w") as f:
            json.dump(file, f, indent=2, ensure_ascii=False)
    if results:
        print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
