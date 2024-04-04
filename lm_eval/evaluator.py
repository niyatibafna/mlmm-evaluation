import collections
import itertools
import numpy as np
import random
import lm_eval.metrics
import lm_eval.models
import lm_eval.tasks
import lm_eval.base
from lm_eval.utils import positional_deprecated, run_task_tests

import sys
sys.path.append('/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/')
sys.path.append('/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/noisers/')
from noisers.main import apply_noisers
from noisers.main import NOISE_REGISTRY

def noise_llm_inputs(doc, ctx, description, task, noise_classes):
    if not noise_classes:
        return doc, ctx
    
    if isinstance(task, lm_eval.base.MultipleChoiceTask):
        og_doc = doc.copy()
        # Remove suffix of doc["query"] from the ctx
        assert ctx.endswith(doc["query"])
        ctx_wout_query = ctx[:-len(doc["query"])]
        if description:
            # Remove description from ctx_wout_query
            assert ctx_wout_query.startswith(description)
            ctx_wout_query = ctx_wout_query[len(description):]

        # Apply noise to the context if we want to
        ## For now, we only apply noise functions to the query and choices in the doc
        ## (and not the few shot context.)
        ## This is because we are simulating a super low-resource language,
        ## and saying that we have the shots in a related high-resource language.
            
        # ctx_wout_query = noiser_main(ctx_wout_query, all_noise_params)

        # Apply noise to the query
        doc["query"] = apply_noisers(doc["query"], noise_classes)

        # Construct the new context
        ctx = description + ctx_wout_query + doc["query"]
        for i, choice in enumerate(doc["choices"]):
            doc["choices"][i] = apply_noisers(choice, noise_classes)
        

    else:
        raise NotImplementedError(f"Task type {type(task)} not supported yet.")
    
    print(f"PRINTING DOC AFTER NOISING: {doc}")
    print(f"Task: {task}")
    print(f"Original Query: {og_doc['query']}")
    print(f"Noised Query: {doc['query']}")
    print(f"Original Choices: {og_doc['choices']}")
    print(f"Noised Choices: {doc['choices']}")
    print(f"\n\n\n")

    return doc, ctx
    

@positional_deprecated
def open_llm_evaluate(
    model,
    model_args=None,
    tasks=[],
    batch_size=None,
    device=None,
    no_cache=False,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    check_integrity=False,
    decontamination_ngrams_path=None,
    write_out=False,
    output_base_path=None,
    noiser_classes=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param batch_size: int, optional
        Batch size for model
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write details about prompts and logits to json for all tasks
    :param output_base_path: str, optional
        Directory to which detailed eval info will be written. Defaults to present working dir.
    :return
        Dictionary of results
    """
    random.seed(1234)
    np.random.seed(1234)

    assert tasks != [], "No tasks specified"

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        lm = lm_eval.models.get_model(model).create_from_arg_string(
            model_args, {"batch_size": batch_size, "device": device}
        )
    else:
        assert isinstance(model, lm_eval.base.LM)
        lm = model

    task_dict = lm_eval.tasks.get_task_dict(tasks)

    if check_integrity:
        print('| Check integrity of tasks')
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        description_dict=description_dict,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        output_base_path=output_base_path,
        noiser_classes=noiser_classes,
    )

    # add info about the model and few shot config
    results["config"] = {
        "model": model,
        "model_args": model_args,
        "batch_size": batch_size,
        "device": device,
        "no_cache": no_cache,
        "limit": limit,
        "bootstrap_iters": bootstrap_iters,
        "description_dict": description_dict,
    }

    return results


decontaminate_suffix = "_decontaminate"


@positional_deprecated
def evaluate(
    lm,
    task_dict,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    decontamination_ngrams_path=None,
    write_out=False,
    output_base_path=None,
    noiser_classes=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param write_out: bool
        If True, write all prompts, logits and metrics to json for offline analysis
    :param output_base_path: str, optional
        Directory to which detailed eval info will be written. Defaults to present working dir
    :return
        Dictionary of results
    """
    # TODO: completely refactor this entire function to not be a huge mess, ideally breaking it down into smaller pieces

    # TODO: todo: implement proper description-providing system

    decontaminate = decontamination_ngrams_path is not None

    task_dict_items = [
        (name, task, task.NUM_FEW_SHOT)
        for name, task in task_dict.items()
        if (task.has_validation_docs() or task.has_test_docs())
    ]

    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)

    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    overlaps = collections.defaultdict(list)  # {task_name: contaminated_docs}

    # If we ever run into issues where the eval tasks don't fit in memory and we can't afford a machine with bigger
    # memory, we can always modify this plumbing to support that, but I didn't want to include it just yet because
    # over-engineering is bad (or we could make it write the requests to disk and then read them back out again
    #  - probably using an sqlite db because of all the moving parts we have

    # TODO: we need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable
    docs = {}
    write_out_info = {}

    docs_for_decontamination = collections.defaultdict(list)

    # get lists of each type of request
    for task_name, task, num_fewshot in task_dict_items:
        versions[task_name] = task.VERSION
        # default to test doc, fall back to val doc if validation unavailable
        # TODO: the test-fallback-to-val system isn't final, we should revisit it at some point
        if task.has_test_docs():
            print('Using test docs for task "{}"'.format(task_name))
            task_doc_func = task.test_docs
            task_set = "test"  # Required for caching in the decontamination
        elif task.has_validation_docs():
            print('Using validation docs for task "{}"'.format(task_name))
            task_set = "val"  # Required for caching in the decontamination
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")

        # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
        task_docs = list(task_doc_func())
        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(task_docs)

        print(f"Task: {task_name}; number of docs: {len(task_docs)}")

        if write_out:
            prompt_details = []

        description = (
            description_dict[task_name]
            if description_dict and task_name in description_dict
            else ""
        )
        if limit is not None:
            limit = int(len(task_docs) * limit) if limit < 1.0 else int(limit)

        for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):
            if decontaminate and task.should_decontaminate():
                docs_for_decontamination[(task_name, task_set)].append(
                    task.doc_to_decontamination_query(doc)
                )

            # The following function chooses shots and constructs the context
            # as ctx = description + labeled_examples + example
            # where example is doc["query"]
            ctx = task.fewshot_context(
                doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description
            )

            # Apply noise to the LLM inputs:
            doc, ctx = noise_llm_inputs(doc, ctx, description, task, noiser_classes)
            
            docs[(task_name, doc_id)] = doc

            # Construct the requests
            ## The following code simply formats the Request object for the task,
            ## e.g. for MultipleChoiceTask, it constructs a Request object containing a list
            ## with the query and each choice.
            ## Like this:
            ## [Req_loglikelihood('Question: जब कोई विवादक खंडन करते समय वास्तविक या झूठे तौर पर खंडन करने की क्षमता की कमी के कारण भ्रम उत्पन्न करता है, तब वह विवादक गलती कर चुका हो सकता है\nChoices:\nA. बेकरी वाला खेल\nB. दया की अपील\nC. व्यक्ति के खिलाफ विवाद\nD. खंडन की अज्ञानता\nAnswer: खंडन की अज्ञानता\n\nQuestion: निष्पक्ष गैर-कार्यकारी बोर्ड सदस्यों की स्वतंत्रता सुनिश्चित करने के लिए, कुछ चरण उठाए जा सकते हैं, जिसमें शामिल हैं _______ कंपनी से, एक _________ समय अवधि के लिए नियुक्ति, साथ ही साथ _________ की नियुक्ति।\nChoices:\nA. बाहर, सीमित, स्वतंत्र रूप से\nB. भीतर, सीमित, अंतरालवार\nC. बाहर, असीमित, अंतरालवार\nD. भीतर, असीमित, स्वतंत्र रूप से\nAnswer: बाहर, सीमित, स्वतंत्र रूप से\n\nQuestion: 2020 की मेडिकल ज्ञान के अनुसार एक रोगी का कैनुलेट करने के लिए आप कितनी कोशिशें करना चाहिए, फिर आप सीनियर कॉलीग को जॉब पास कर सकते हैं?\nChoices:\nA. 4\nB. 3\nC. 2\nD. 1\nAnswer: 2\n\nQuestion: केंद्रीय प्रवृत्ति को मापने के तीन तरीके होते हैं: मीन, मीडियन और मोड। उनके बारें में आपके ज्ञान के आधार पर, मोड क्या होता है?\nChoices:\nA. मीडियन से में अति-एक्सट्रीम स्कोरों के प्रति कम संवेदनशील होता है\nB. ढीले वितरण के लिए अधिक उपयोगी होता है\nC. अतिमाहत्त्व वाले मूल्यों और अधिक ढीले वितरणों वाला प्रदर्शनशील होता है\nD. सबसे अधिक बार आने वाली संख्या\nAnswer: सबसे अधिक बार आने वाली संख्या\n\nQuestion: प्ल्यूरा\nChoices:\nA. संवेदनाशील संवहन नहीं हैं।\nB. 2 मिमी अंतराल द्वारा अलग होते हैं।\nC. गर्दन तक फैलते हैं।\nD. श्वसन एपिथेलियम से बने होते हैं।\nAnswer: गर्दन तक फैलते हैं।\n\nQuestion: 2019 में निम्नलिखित वाक्यों के कौन से दोनों सत्य हैं?\nChoices:\nA. लोग अपने भविष्य और अपनी राष्ट्र या दुनिया के भविष्य के बारे में आशावादी होते हैं।\nB. लोग अपने भविष्य के बारे में आशावादी होते हैं लेकिन अपनी राष्ट्र या दुनिया के भविष्य के बारे में निराशावादी होते हैं।\nC. लोग अपने भविष्य के बारे में निराशावादी होते हैं लेकिन अपनी राष्ट्र या दुनिया के भविष्य के बारे में आशावादी होते हैं।\nD. लोग अपने भविष्य और अपनी राष्ट्र या दुनिया के भविष्य के बारे में निराशावादी होते हैं।\nAnswer: लोग अपने भविष्य के बारे में आशावादी होते हैं लेकिन अपनी राष्ट्र या दुनिया के भविष्य के बारे में निराशावादी होते हैं।\n\nQuestion: एक वास्तविक 2x2 रीयल मैट्रिक्स A हो। निम्नलिखित में से कौन सा कथन सही होगा?\r\nI. A^2 के सभी प्रविष्टियों का अमैश्वर्य होगा।\r\nII. A^2 का निर्णायक अमैश्वर्य है।\r\nIII. अगर A के दो अलग-अलग इगेनवैल्यू हैं तो A^2 के दो अलग-अलग इगेनवैल्यू होंगे।\nChoices:\nA. केवल I\nB. केवल II\nC. केवल III\nD. केवल II और III\nAnswer: केवल II\n\nQuestion: यूनाइटेड स्टेट्स और दुनिया के बीच संबंधों से जुड़े नीति निर्णयों के क्षेत्र को विदेश नीति के नाम से जाना जाता है\nChoices:\nA. आतंकवाद नीति।\nB. आर्थिक नीति।\nC. विदेश नीति।\nD. अंतरराष्ट्रीय नीति।\nAnswer: विदेश नीति।\n\nQuestion: बर्गर (Berger) (1963) सामाजिक वास्तविकता के लिए किस मेटाफॉर का वर्णन करते हैं?\nChoices:\nA. एक मेला का सवारी\nB. एक सर्कस\nC. एक कठपुतली थिएटर\nD. एक बैलेट\nAnswer: एक कठपुतली थिएटर\n\nQuestion: मेयोसिस का वह चरण जिसमें क्रोमोसोम पैयर होते हैं और क्रॉस ओवर होती है, है:\nChoices:\nA. प्रोफेज I\nB. मेटाफेज I\nC. प्रोफेज II\nD. मेटाफेज II\nAnswer: प्रोफेज I\n\nQuestion: हायड बोन का भ्रूणात्मक मूल क्या है?\nChoices:\nA. प्रथम फैरिंजिअल आर्च\nB. प्रथम और दूसरे फैरिंजिअल आर्च\nC. दूसरे फैरिंजिअल आर्च\nD. दूसरे और तीसरे फैरिंजिअल आर्च\nAnswer: दूसरे और तीसरे फैरिंजिअल आर्च\n\nQuestion: CSR में अंगंगवाद के साथ संबंधित कुछ नैतिक विवाद हैं: बाह्यताएं, जो कंपनियों के पास शक्ति के संबंध में होती हैं, और व्यापार और समाज की ________।\nChoices:\nA. बाहरीता, शक्ति, स्वतंत्रता\nB. प्रचार, अस्थायी संसाधन, सहयोगी आवश्यकता\nC. प्रचार, शक्ति, स्वतंत्रता\nD. बाह्यताएं, शक्ति, सहयोगी आवश्यकता\nAnswer: बाह्यताएं, शक्ति, सहयोगी आवश्यकता\n\nQuestion: नीचे दिए गए प्रोग्राम में, x का प्रारंभिक मूल्य 5 है और y का प्रारंभिक मूल्य 10 है।\nIF (X < O)\n{\n  डिस्प्ले करें ("Foxtrot")\n}\nELSE\n{\n  IF (X > y)\n  {\n     डिस्प्ले करें ("Hotel")\n  }\n  ELSE\n  {\n     IF (y > O)\n     {\n        डिस्प्ले करें ("November")\n     }\n     ELSE\n     {\n        डिस्प्ले करें ("Yankee")\n     }\n  }\n}\n\nप्रोग्राम चलाने का परिणाम क्या होगा?\nChoices:\nA. Foxtrot\nB. Hotel\nC. November\nD. Yankee\nAnswer: November\n\nQuestion: एक नवजात के जीनेटिक टेस्ट में, एक दुर्लभ जीनेटिक विकार पाया गया है जिसका X-लिंक्ड रीसेसिव संचार होता है। निम्नलिखित कथनों में से कौन सा विकार के पेडिग्री के संबंध में संभवतः सत्य है?\nChoices:\nA. मातृसूची के सभी वंशज इस विकार से प्रभावित होंगे।\nB. इस परिवार में महिलाएं लगभग दोगुना अधिक प्रभावित होंगी जबकि पुरुषों को।\nC. एक प्रभावित पुरुष की सभी बेटियाँ प्रभावित होंगी।\nD. पुरुषों और महिलाओं में प्रभावित होने का समान वितरण होगा।\nAnswer: एक प्रभावित पुरुष की सभी बेटियाँ प्रभावित होंगी।\n\nQuestion: निम्नलिखित थर्मोडायनामिक प्रक्रियाओं में से कौनसी प्रक्रिया में आईडीएल गैस के आंतरिक ऊर्जा का वृद्धि गैस में जोड़ी गई ऊष्मा के बराबर होता है?\nChoices:\nA. निरंतर तापमान\nB. निरंतर वास\nC. निरंतर दबाव\nD. अदिबद्ध\nAnswer: निरंतर वास\n\nQuestion: निम्नलिखित पूर्ण प्रस्तावों के लिए एक पूर्ण सत्यता सारणी बनाएं। फिर, सत्यता सारणियों का उपयोग करके, यह निश्चित करें कि क्या वाक्य तार्किक रूप से समान हैं या विरोधाभासी हैं। अगर ना तो यह तय करें कि वे सुसंगत हैं या असंगत हैं। आपके उत्तरों का न्याय दिलाएं।\nE ⊃ (F · E) और ~E · F\nChoices:\nA. तार्किक रूप से समान हैं\nB. विरोधाभासी हैं\nC. तार्किक रूप से समान नहीं हैं या विरोधाभासी नहीं हैं, लेकिन सुसंगत हैं\nD. असंगत हैं\nAnswer: तार्किक रूप से समान नहीं हैं या विरोधाभासी नहीं हैं, लेकिन सुसंगत हैं\n\nQuestion: निम्नलिखित वाक्य के सबसे अच्छे चिह्नीकरण के लिए PL के दिए गए सूत्रों में से कौन सा है? \nकछुए लंबी उम्र तक जीवित रहते हैं और खुश जीव होते हैं, जब तक कि वे घायल न हो जाएँ।\nChoices:\nA. (L • H) ≡ I\nB. (L • H) ∨ I\nC. L • (H ∨ I)\nD. L • (H ⊃ R)\nAnswer: (L • H) ∨ I\n\nQuestion: इनमें से कौन सा सूची वर्ग-14 तत्वों के हाइड्राइडों को थर्मल स्थिरता के क्रम में सबसे कम से सबसे उच्च तक के क्रम में दर्शाता है?\nChoices:\nA. PbH4 < SnH4 < GeH4 < SiH4 < CH4\nB. PbH4 < SnH4 < CH4 < GeH4 < SiH4\nC. CH4 < SiH4 < GeH4 < SnH4 < PbH4\nD. CH4 < PbH4 < GeH4 < SnH4 < SiH4\nAnswer: PbH4 < SnH4 < GeH4 < SiH4 < CH4\n\nQuestion: सामान्य ऑटोमोबाइल में कितने एक्सल होते हैं?\nChoices:\nA. एक\nB. दो\nC. चार\nD. आठ\nAnswer: दो\n\nQuestion: मेंटोसिस और मेयोसिस में क्रोमोसोमों के अलगाव के लिए आवश्यक डीएनए सीक्वेंस कौनसा है?\nChoices:\nA. टेलोमेर\nB. सेन्ट्रोमियर\nC. न्यूक्लिओसोम्स\nD. स्प्लाइसोसोम्स\nAnswer: सेन्ट्रोमियर\n\nQuestion: एक राज्य की बाँधने की सहमति को कैसे व्यक्त किया जा सकता है?\nChoices:\nA. किसी विश्वसनीयता से एक राज्य की सहमति तय की जाती है\nB. एक संधि से एक राज्य की सहमति हस्ताक्षर, विश्वसनीयता, स्वीकृति, मंजूरी या शामिल होकर व्यक्त की जा सकती है।\nC. किसी संधि से एक राज्य की सहमति हस्ताक्षर द्वारा व्यक्त की जाती है\nD. एक राज्य की सहमति कार्य में आने वाले कुछ भी माध्यम से व्यक्त की जाती है\nAnswer: एक संधि से एक राज्य की सहमति हस्ताक्षर, विश्वसनीयता, स्वीकृति, मंजूरी या शामिल होकर व्यक्त की जा सकती है।\n\nQuestion: इन तीन तत्वों में हर न्यूक्लियॉन के लिए सबसे अधिक भार किसमें है?\nChoices:\nA. हाइड्रोजन\nB. लोहा\nC. यूरेनियम\nD. हर तत्व में समान\nAnswer: हाइड्रोजन\n\nQuestion: एक सौ साल पहले, आपकी दादी की दादी ने 5% वार्षिक ब्याज पर $100 का निवेश किया था। आज निवेश कितना मूल्यवान हो गया है?\nChoices:\nA. $13,000\nB. $600\nC. $15,000\nD. $28,000\nAnswer: $13,000\n\nQuestion: उपनिषदों का विशेषता क्या है?\nChoices:\nA. रीति-पाठ\nB. दर्शनिक पाठ\nC. स्तुति\nD. उत्पत्ति कहानियां\nAnswer: दर्शनिक पाठ\n\nQuestion: प्रेडिकेट तर्क में सर्वश्रेष्ठ अनुवाद का चयन करें: मंगल ग्रह पर कोई लोग गाड़ी नहीं चलाते।\nChoices:\nA. ~Pd\nB. (∀x)(Px ∨ ~Dx)\nC. (∀x)(Px ⊃ ~Dx)\nD. ~Dp\nAnswer: (∀x)(Px ⊃ ~Dx)\n\nuestion: 2020 के अमेरिकी सामान्य नैतिक मानकों के अनुसार, इन दो स्थितियों में से कौन सी स्थिति में मुख्य पात्र (जो मैं / मुझे / मेरा उपयोग करता है) नैतिक रूप से स्पष्ट रूप से गलत कुछ करता है? स्थिति 1 | मैं अपनी पत्नी को घर ले जाने से पहले अपनी गर्लफ्रेंड के पास जाकर कुछ फूल दे दिए। स्थिति 2 | मैंने कॉपी और पेस्ट कर लिया ताकि मैं इंटरनेट पर मिली स्रोतों के साथ अपना पेपर लिख सकूँ।\nChoices:\nA. गलत, गलत\nB. गलत, गलत नहीं है\nC. गलत नहीं है, गलत\nD. गलत नहीं है, गलत नहीं है\nAnswer:', ' NOISE MARK गलत, गलत')[0]
            ## , Req_loglikelihood('Question: जब कोई विवादक खंडन करते समय वास्तविक या झूठे तौर पर खंडन करने की क्षमता की कमी के कारण भ्रम उत्पन्न करता है, तब वह विवादक गलती कर चुका हो सकता है\nChoices:\nA. बेकरी वाला खेल\nB. दया की अपील\nC. व्यक्ति के खिलाफ विवाद\nD. खंडन की अज्ञानता\nAnswer: खंडन की अज्ञानता\n\nQuestion: निष्पक्ष गैर-कार्यकारी बोर्ड सदस्यों की स्वतंत्रता सुनिश्चित करने के लिए, कुछ चरण उठाए जा सकते हैं, जिसमें शामिल हैं _______ कंपनी से, एक _________ समय अवधि के लिए नियुक्ति, साथ ही साथ _________ की नियुक्ति।\nChoices:\nA. बाहर, सीमित, स्वतंत्र रूप से\nB. भीतर, सीमित, अंतरालवार\nC. बाहर, असीमित, अंतरालवार\nD. भीतर, असीमित, स्वतंत्र रूप से\nAnswer: बाहर, सीमित, स्वतंत्र रूप से\n\nQuestion: 2020 की मेडिकल ज्ञान के अनुसार एक रोगी का कैनुलेट करने के लिए आप कितनी कोशिशें करना चाहिए, फिर आप सीनियर कॉलीग को जॉब पास कर सकते हैं?\nChoices:\nA. 4\nB. 3\nC. 2\nD. 1\nAnswer: 2\n\nQuestion: केंद्रीय प्रवृत्ति को मापने के तीन तरीके होते हैं: मीन, मीडियन और मोड। उनके बारें में आपके ज्ञान के आधार पर, मोड क्या होता है?\nChoices:\nA. मीडियन से में अति-एक्सट्रीम स्कोरों के प्रति कम संवेदनशील होता है\nB. ढीले वितरण के लिए अधिक उपयोगी होता है\nC. अतिमाहत्त्व वाले मूल्यों और अधिक ढीले वितरणों वाला प्रदर्शनशील होता है\nD. सबसे अधिक बार आने वाली संख्या\nAnswer: सबसे अधिक बार आने वाली संख्या\n\nQuestion: प्ल्यूरा\nChoices:\nA. संवेदनाशील संवहन नहीं हैं।\nB. 2 मिमी अंतराल द्वारा अलग होते हैं।\nC. गर्दन तक फैलते हैं।\nD. श्वसन एपिथेलियम से बने होते हैं।\nAnswer: गर्दन तक फैलते हैं।\n\nQuestion: 2019 में निम्नलिखित वाक्यों के कौन से दोनों सत्य हैं?\nChoices:\nA. लोग अपने भविष्य और अपनी राष्ट्र या दुनिया के भविष्य के बारे में आशावादी होते हैं।\nB. लोग अपने भविष्य के बारे में आशावादी होते हैं लेकिन अपनी राष्ट्र या दुनिया के भविष्य के बारे में निराशावादी होते हैं।\nC. लोग अपने भविष्य के बारे में निराशावादी होते हैं लेकिन अपनी राष्ट्र या दुनिया के भविष्य के बारे में आशावादी होते हैं।\nD. लोग अपने भविष्य और अपनी राष्ट्र या दुनिया के भविष्य के बारे में निराशावादी होते हैं।\nAnswer: लोग अपने भविष्य के बारे में आशावादी होते हैं लेकिन अपनी राष्ट्र या दुनिया के भविष्य के बारे में निराशावादी होते हैं।\n\nQuestion: एक वास्तविक 2x2 रीयल मैट्रिक्स A हो। निम्नलिखित में से कौन सा कथन सही होगा?\r\nI. A^2 के सभी प्रविष्टियों का अमैश्वर्य होगा।\r\nII. A^2 का निर्णायक अमैश्वर्य है।\r\nIII. अगर A के दो अलग-अलग इगेनवैल्यू हैं तो A^2 के दो अलग-अलग इगेनवैल्यू होंगे।\nChoices:\nA. केवल I\nB. केवल II\nC. केवल III\nD. केवल II और III\nAnswer: केवल II\n\nQuestion: यूनाइटेड स्टेट्स और दुनिया के बीच संबंधों से जुड़े नीति निर्णयों के क्षेत्र को विदेश नीति के नाम से जाना जाता है\nChoices:\nA. आतंकवाद नीति।\nB. आर्थिक नीति।\nC. विदेश नीति।\nD. अंतरराष्ट्रीय नीति।\nAnswer: विदेश नीति।\n\nQuestion: बर्गर (Berger) (1963) सामाजिक वास्तविकता के लिए किस मेटाफॉर का वर्णन करते हैं?\nChoices:\nA. एक मेला का सवारी\nB. एक सर्कस\nC. एक कठपुतली थिएटर\nD. एक बैलेट\nAnswer: एक कठपुतली थिएटर\n\nQuestion: मेयोसिस का वह चरण जिसमें क्रोमोसोम पैयर होते हैं और क्रॉस ओवर होती है, है:\nChoices:\nA. प्रोफेज I\nB. मेटाफेज I\nC. प्रोफेज II\nD. मेटाफेज II\nAnswer: प्रोफेज I\n\nQuestion: हायड बोन का भ्रूणात्मक मूल क्या है?\nChoices:\nA. प्रथम फैरिंजिअल आर्च\nB. प्रथम और दूसरे फैरिंजिअल आर्च\nC. दूसरे फैरिंजिअल आर्च\nD. दूसरे और तीसरे फैरिंजिअल आर्च\nAnswer: दूसरे और तीसरे फैरिंजिअल आर्च\n\nQuestion: CSR में अंगंगवाद के साथ संबंधित कुछ नैतिक विवाद हैं: बाह्यताएं, जो कंपनियों के पास शक्ति के संबंध में होती हैं, और व्यापार और समाज की ________।\nChoices:\nA. बाहरीता, शक्ति, स्वतंत्रता\nB. प्रचार, अस्थायी संसाधन, सहयोगी आवश्यकता\nC. प्रचार, शक्ति, स्वतंत्रता\nD. बाह्यताएं, शक्ति, सहयोगी आवश्यकता\nAnswer: बाह्यताएं, शक्ति, सहयोगी आवश्यकता\n\nQuestion: नीचे दिए गए प्रोग्राम में, x का प्रारंभिक मूल्य 5 है और y का प्रारंभिक मूल्य 10 है।\nIF (X < O)\n{\n  डिस्प्ले करें ("Foxtrot")\n}\nELSE\n{\n  IF (X > y)\n  {\n     डिस्प्ले करें ("Hotel")\n  }\n  ELSE\n  {\n     IF (y > O)\n     {\n        डिस्प्ले करें ("November")\n     }\n     ELSE\n     {\n        डिस्प्ले करें ("Yankee")\n     }\n  }\n}\n\nप्रोग्राम चलाने का परिणाम क्या होगा?\nChoices:\nA. Foxtrot\nB. Hotel\nC. November\nD. Yankee\nAnswer: November\n\nQuestion: एक नवजात के जीनेटिक टेस्ट में, एक दुर्लभ जीनेटिक विकार पाया गया है जिसका X-लिंक्ड रीसेसिव संचार होता है। निम्नलिखित कथनों में से कौन सा विकार के पेडिग्री के संबंध में संभवतः सत्य है?\nChoices:\nA. मातृसूची के सभी वंशज इस विकार से प्रभावित होंगे।\nB. इस परिवार में महिलाएं लगभग दोगुना अधिक प्रभावित होंगी जबकि पुरुषों को।\nC. एक प्रभावित पुरुष की सभी बेटियाँ प्रभावित होंगी।\nD. पुरुषों और महिलाओं में प्रभावित होने का समान वितरण होगा।\nAnswer: एक प्रभावित पुरुष की सभी बेटियाँ प्रभावित होंगी।\n\nQuestion: निम्नलिखित थर्मोडायनामिक प्रक्रियाओं में से कौनसी प्रक्रिया में आईडीएल गैस के आंतरिक ऊर्जा का वृद्धि गैस में जोड़ी गई ऊष्मा के बराबर होता है?\nChoices:\nA. निरंतर तापमान\nB. निरंतर वास\nC. निरंतर दबाव\nD. अदिबद्ध\nAnswer: निरंतर वास\n\nQuestion: निम्नलिखित पूर्ण प्रस्तावों के लिए एक पूर्ण सत्यता सारणी बनाएं। फिर, सत्यता सारणियों का उपयोग करके, यह निश्चित करें कि क्या वाक्य तार्किक रूप से समान हैं या विरोधाभासी हैं। अगर ना तो यह तय करें कि वे सुसंगत हैं या असंगत हैं। आपके उत्तरों का न्याय दिलाएं।\nE ⊃ (F · E) और ~E · F\nChoices:\nA. तार्किक रूप से समान हैं\nB. विरोधाभासी हैं\nC. तार्किक रूप से समान नहीं हैं या विरोधाभासी नहीं हैं, लेकिन सुसंगत हैं\nD. असंगत हैं\nAnswer: तार्किक रूप से समान नहीं हैं या विरोधाभासी नहीं हैं, लेकिन सुसंगत हैं\n\nQuestion: निम्नलिखित वाक्य के सबसे अच्छे चिह्नीकरण के लिए PL के दिए गए सूत्रों में से कौन सा है? \nकछुए लंबी उम्र तक जीवित रहते हैं और खुश जीव होते हैं, जब तक कि वे घायल न हो जाएँ।\nChoices:\nA. (L • H) ≡ I\nB. (L • H) ∨ I\nC. L • (H ∨ I)\nD. L • (H ⊃ R)\nAnswer: (L • H) ∨ I\n\nQuestion: इनमें से कौन सा सूची वर्ग-14 तत्वों के हाइड्राइडों को थर्मल स्थिरता के क्रम में सबसे कम से सबसे उच्च तक के क्रम में दर्शाता है?\nChoices:\nA. PbH4 < SnH4 < GeH4 < SiH4 < CH4\nB. PbH4 < SnH4 < CH4 < GeH4 < SiH4\nC. CH4 < SiH4 < GeH4 < SnH4 < PbH4\nD. CH4 < PbH4 < GeH4 < SnH4 < SiH4\nAnswer: PbH4 < SnH4 < GeH4 < SiH4 < CH4\n\nQuestion: सामान्य ऑटोमोबाइल में कितने एक्सल होते हैं?\nChoices:\nA. एक\nB. दो\nC. चार\nD. आठ\nAnswer: दो\n\nQuestion: मेंटोसिस और मेयोसिस में क्रोमोसोमों के अलगाव के लिए आवश्यक डीएनए सीक्वेंस कौनसा है?\nChoices:\nA. टेलोमेर\nB. सेन्ट्रोमियर\nC. न्यूक्लिओसोम्स\nD. स्प्लाइसोसोम्स\nAnswer: सेन्ट्रोमियर\n\nQuestion: एक राज्य की बाँधने की सहमति को कैसे व्यक्त किया जा सकता है?\nChoices:\nA. किसी विश्वसनीयता से एक राज्य की सहमति तय की जाती है\nB. एक संधि से एक राज्य की सहमति हस्ताक्षर, विश्वसनीयता, स्वीकृति, मंजूरी या शामिल होकर व्यक्त की जा सकती है।\nC. किसी संधि से एक राज्य की सहमति हस्ताक्षर द्वारा व्यक्त की जाती है\nD. एक राज्य की सहमति कार्य में आने वाले कुछ भी माध्यम से व्यक्त की जाती है\nAnswer: एक संधि से एक राज्य की सहमति हस्ताक्षर, विश्वसनीयता, स्वीकृति, मंजूरी या शामिल होकर व्यक्त की जा सकती है।\n\nQuestion: इन तीन तत्वों में हर न्यूक्लियॉन के लिए सबसे अधिक भार किसमें है?\nChoices:\nA. हाइड्रोजन\nB. लोहा\nC. यूरेनियम\nD. हर तत्व में समान\nAnswer: हाइड्रोजन\n\nQuestion: एक सौ साल पहले, आपकी दादी की दादी ने 5% वार्षिक ब्याज पर $100 का निवेश किया था। आज निवेश कितना मूल्यवान हो गया है?\nChoices:\nA. $13,000\nB. $600\nC. $15,000\nD. $28,000\nAnswer: $13,000\n\nQuestion: उपनिषदों का विशेषता क्या है?\nChoices:\nA. रीति-पाठ\nB. दर्शनिक पाठ\nC. स्तुति\nD. उत्पत्ति कहानियां\nAnswer: दर्शनिक पाठ\n\nQuestion: प्रेडिकेट तर्क में सर्वश्रेष्ठ अनुवाद का चयन करें: मंगल ग्रह पर कोई लोग गाड़ी नहीं चलाते।\nChoices:\nA. ~Pd\nB. (∀x)(Px ∨ ~Dx)\nC. (∀x)(Px ⊃ ~Dx)\nD. ~Dp\nAnswer: (∀x)(Px ⊃ ~Dx)\n\nuestion: 2020 के अमेरिकी सामान्य नैतिक मानकों के अनुसार, इन दो स्थितियों में से कौन सी स्थिति में मुख्य पात्र (जो मैं / मुझे / मेरा उपयोग करता है) नैतिक रूप से स्पष्ट रूप से गलत कुछ करता है? स्थिति 1 | मैं अपनी पत्नी को घर ले जाने से पहले अपनी गर्लफ्रेंड के पास जाकर कुछ फूल दे दिए। स्थिति 2 | मैंने कॉपी और पेस्ट कर लिया ताकि मैं इंटरनेट पर मिली स्रोतों के साथ अपना पेपर लिख सकूँ।\nChoices:\nA. गलत, गलत\nB. गलत, गलत नहीं है\nC. गलत नहीं है, गलत\nD. गलत नहीं है, गलत नहीं है\nAnswer:', ' NOISE MARK गलत, गलत नहीं है')[0]
            ## , Req_loglikelihood('Question: जब कोई विवादक खंडन करते समय वास्तविक या झूठे तौर पर खंडन करने की क्षमता की कमी के कारण भ्रम उत्पन्न करता है, तब वह विवादक गलती कर चुका हो सकता है\nChoices:\nA. बेकरी वाला खेल\nB. दया की अपील\nC. व्यक्ति के खिलाफ विवाद\nD. खंडन की अज्ञानता\nAnswer: खंडन की अज्ञानता\n\nQuestion: निष्पक्ष गैर-कार्यकारी बोर्ड सदस्यों की स्वतंत्रता सुनिश्चित करने के लिए, कुछ चरण उठाए जा सकते हैं, जिसमें शामिल हैं _______ कंपनी से, एक _________ समय अवधि के लिए नियुक्ति, साथ ही साथ _________ की नियुक्ति।\nChoices:\nA. बाहर, सीमित, स्वतंत्र रूप से\nB. भीतर, सीमित, अंतरालवार\nC. बाहर, असीमित, अंतरालवार\nD. भीतर, असीमित, स्वतंत्र रूप से\nAnswer: बाहर, सीमित, स्वतंत्र रूप से\n\nQuestion: 2020 की मेडिकल ज्ञान के अनुसार एक रोगी का कैनुलेट करने के लिए आप कितनी कोशिशें करना चाहिए, फिर आप सीनियर कॉलीग को जॉब पास कर सकते हैं?\nChoices:\nA. 4\nB. 3\nC. 2\nD. 1\nAnswer: 2\n\nQuestion: केंद्रीय प्रवृत्ति को मापने के तीन तरीके होते हैं: मीन, मीडियन और मोड। उनके बारें में आपके ज्ञान के आधार पर, मोड क्या होता है?\nChoices:\nA. मीडियन से में अति-एक्सट्रीम स्कोरों के प्रति कम संवेदनशील होता है\nB. ढीले वितरण के लिए अधिक उपयोगी होता है\nC. अतिमाहत्त्व वाले मूल्यों और अधिक ढीले वितरणों वाला प्रदर्शनशील होता है\nD. सबसे अधिक बार आने वाली संख्या\nAnswer: सबसे अधिक बार आने वाली संख्या\n\nQuestion: प्ल्यूरा\nChoices:\nA. संवेदनाशील संवहन नहीं हैं।\nB. 2 मिमी अंतराल द्वारा अलग होते हैं।\nC. गर्दन तक फैलते हैं।\nD. श्वसन एपिथेलियम से बने होते हैं।\nAnswer: गर्दन तक फैलते हैं।\n\nQuestion: 2019 में निम्नलिखित वाक्यों के कौन से दोनों सत्य हैं?\nChoices:\nA. लोग अपने भविष्य और अपनी राष्ट्र या दुनिया के भविष्य के बारे में आशावादी होते हैं।\nB. लोग अपने भविष्य के बारे में आशावादी होते हैं लेकिन अपनी राष्ट्र या दुनिया के भविष्य के बारे में निराशावादी होते हैं।\nC. लोग अपने भविष्य के बारे में निराशावादी होते हैं लेकिन अपनी राष्ट्र या दुनिया के भविष्य के बारे में आशावादी होते हैं।\nD. लोग अपने भविष्य और अपनी राष्ट्र या दुनिया के भविष्य के बारे में निराशावादी होते हैं।\nAnswer: लोग अपने भविष्य के बारे में आशावादी होते हैं लेकिन अपनी राष्ट्र या दुनिया के भविष्य के बारे में निराशावादी होते हैं।\n\nQuestion: एक वास्तविक 2x2 रीयल मैट्रिक्स A हो। निम्नलिखित में से कौन सा कथन सही होगा?\r\nI. A^2 के सभी प्रविष्टियों का अमैश्वर्य होगा।\r\nII. A^2 का निर्णायक अमैश्वर्य है।\r\nIII. अगर A के दो अलग-अलग इगेनवैल्यू हैं तो A^2 के दो अलग-अलग इगेनवैल्यू होंगे।\nChoices:\nA. केवल I\nB. केवल II\nC. केवल III\nD. केवल II और III\nAnswer: केवल II\n\nQuestion: यूनाइटेड स्टेट्स और दुनिया के बीच संबंधों से जुड़े नीति निर्णयों के क्षेत्र को विदेश नीति के नाम से जाना जाता है\nChoices:\nA. आतंकवाद नीति।\nB. आर्थिक नीति।\nC. विदेश नीति।\nD. अंतरराष्ट्रीय नीति।\nAnswer: विदेश नीति।\n\nQuestion: बर्गर (Berger) (1963) सामाजिक वास्तविकता के लिए किस मेटाफॉर का वर्णन करते हैं?\nChoices:\nA. एक मेला का सवारी\nB. एक सर्कस\nC. एक कठपुतली थिएटर\nD. एक बैलेट\nAnswer: एक कठपुतली थिएटर\n\nQuestion: मेयोसिस का वह चरण जिसमें क्रोमोसोम पैयर होते हैं और क्रॉस ओवर होती है, है:\nChoices:\nA. प्रोफेज I\nB. मेटाफेज I\nC. प्रोफेज II\nD. मेटाफेज II\nAnswer: प्रोफेज I\n\nQuestion: हायड बोन का भ्रूणात्मक मूल क्या है?\nChoices:\nA. प्रथम फैरिंजिअल आर्च\nB. प्रथम और दूसरे फैरिंजिअल आर्च\nC. दूसरे फैरिंजिअल आर्च\nD. दूसरे और तीसरे फैरिंजिअल आर्च\nAnswer: दूसरे और तीसरे फैरिंजिअल आर्च\n\nQuestion: CSR में अंगंगवाद के साथ संबंधित कुछ नैतिक विवाद हैं: बाह्यताएं, जो कंपनियों के पास शक्ति के संबंध में होती हैं, और व्यापार और समाज की ________।\nChoices:\nA. बाहरीता, शक्ति, स्वतंत्रता\nB. प्रचार, अस्थायी संसाधन, सहयोगी आवश्यकता\nC. प्रचार, शक्ति, स्वतंत्रता\nD. बाह्यताएं, शक्ति, सहयोगी आवश्यकता\nAnswer: बाह्यताएं, शक्ति, सहयोगी आवश्यकता\n\nQuestion: नीचे दिए गए प्रोग्राम में, x का प्रारंभिक मूल्य 5 है और y का प्रारंभिक मूल्य 10 है।\nIF (X < O)\n{\n  डिस्प्ले करें ("Foxtrot")\n}\nELSE\n{\n  IF (X > y)\n  {\n     डिस्प्ले करें ("Hotel")\n  }\n  ELSE\n  {\n     IF (y > O)\n     {\n        डिस्प्ले करें ("November")\n     }\n     ELSE\n     {\n        डिस्प्ले करें ("Yankee")\n     }\n  }\n}\n\nप्रोग्राम चलाने का परिणाम क्या होगा?\nChoices:\nA. Foxtrot\nB. Hotel\nC. November\nD. Yankee\nAnswer: November\n\nQuestion: एक नवजात के जीनेटिक टेस्ट में, एक दुर्लभ जीनेटिक विकार पाया गया है जिसका X-लिंक्ड रीसेसिव संचार होता है। निम्नलिखित कथनों में से कौन सा विकार के पेडिग्री के संबंध में संभवतः सत्य है?\nChoices:\nA. मातृसूची के सभी वंशज इस विकार से प्रभावित होंगे।\nB. इस परिवार में महिलाएं लगभग दोगुना अधिक प्रभावित होंगी जबकि पुरुषों को।\nC. एक प्रभावित पुरुष की सभी बेटियाँ प्रभावित होंगी।\nD. पुरुषों और महिलाओं में प्रभावित होने का समान वितरण होगा।\nAnswer: एक प्रभावित पुरुष की सभी बेटियाँ प्रभावित होंगी।\n\nQuestion: निम्नलिखित थर्मोडायनामिक प्रक्रियाओं में से कौनसी प्रक्रिया में आईडीएल गैस के आंतरिक ऊर्जा का वृद्धि गैस में जोड़ी गई ऊष्मा के बराबर होता है?\nChoices:\nA. निरंतर तापमान\nB. निरंतर वास\nC. निरंतर दबाव\nD. अदिबद्ध\nAnswer: निरंतर वास\n\nQuestion: निम्नलिखित पूर्ण प्रस्तावों के लिए एक पूर्ण सत्यता सारणी बनाएं। फिर, सत्यता सारणियों का उपयोग करके, यह निश्चित करें कि क्या वाक्य तार्किक रूप से समान हैं या विरोधाभासी हैं। अगर ना तो यह तय करें कि वे सुसंगत हैं या असंगत हैं। आपके उत्तरों का न्याय दिलाएं।\nE ⊃ (F · E) और ~E · F\nChoices:\nA. तार्किक रूप से समान हैं\nB. विरोधाभासी हैं\nC. तार्किक रूप से समान नहीं हैं या विरोधाभासी नहीं हैं, लेकिन सुसंगत हैं\nD. असंगत हैं\nAnswer: तार्किक रूप से समान नहीं हैं या विरोधाभासी नहीं हैं, लेकिन सुसंगत हैं\n\nQuestion: निम्नलिखित वाक्य के सबसे अच्छे चिह्नीकरण के लिए PL के दिए गए सूत्रों में से कौन सा है? \nकछुए लंबी उम्र तक जीवित रहते हैं और खुश जीव होते हैं, जब तक कि वे घायल न हो जाएँ।\nChoices:\nA. (L • H) ≡ I\nB. (L • H) ∨ I\nC. L • (H ∨ I)\nD. L • (H ⊃ R)\nAnswer: (L • H) ∨ I\n\nQuestion: इनमें से कौन सा सूची वर्ग-14 तत्वों के हाइड्राइडों को थर्मल स्थिरता के क्रम में सबसे कम से सबसे उच्च तक के क्रम में दर्शाता है?\nChoices:\nA. PbH4 < SnH4 < GeH4 < SiH4 < CH4\nB. PbH4 < SnH4 < CH4 < GeH4 < SiH4\nC. CH4 < SiH4 < GeH4 < SnH4 < PbH4\nD. CH4 < PbH4 < GeH4 < SnH4 < SiH4\nAnswer: PbH4 < SnH4 < GeH4 < SiH4 < CH4\n\nQuestion: सामान्य ऑटोमोबाइल में कितने एक्सल होते हैं?\nChoices:\nA. एक\nB. दो\nC. चार\nD. आठ\nAnswer: दो\n\nQuestion: मेंटोसिस और मेयोसिस में क्रोमोसोमों के अलगाव के लिए आवश्यक डीएनए सीक्वेंस कौनसा है?\nChoices:\nA. टेलोमेर\nB. सेन्ट्रोमियर\nC. न्यूक्लिओसोम्स\nD. स्प्लाइसोसोम्स\nAnswer: सेन्ट्रोमियर\n\nQuestion: एक राज्य की बाँधने की सहमति को कैसे व्यक्त किया जा सकता है?\nChoices:\nA. किसी विश्वसनीयता से एक राज्य की सहमति तय की जाती है\nB. एक संधि से एक राज्य की सहमति हस्ताक्षर, विश्वसनीयता, स्वीकृति, मंजूरी या शामिल होकर व्यक्त की जा सकती है।\nC. किसी संधि से एक राज्य की सहमति हस्ताक्षर द्वारा व्यक्त की जाती है\nD. एक राज्य की सहमति कार्य में आने वाले कुछ भी माध्यम से व्यक्त की जाती है\nAnswer: एक संधि से एक राज्य की सहमति हस्ताक्षर, विश्वसनीयता, स्वीकृति, मंजूरी या शामिल होकर व्यक्त की जा सकती है।\n\nQuestion: इन तीन तत्वों में हर न्यूक्लियॉन के लिए सबसे अधिक भार किसमें है?\nChoices:\nA. हाइड्रोजन\nB. लोहा\nC. यूरेनियम\nD. हर तत्व में समान\nAnswer: हाइड्रोजन\n\nQuestion: एक सौ साल पहले, आपकी दादी की दादी ने 5% वार्षिक ब्याज पर $100 का निवेश किया था। आज निवेश कितना मूल्यवान हो गया है?\nChoices:\nA. $13,000\nB. $600\nC. $15,000\nD. $28,000\nAnswer: $13,000\n\nQuestion: उपनिषदों का विशेषता क्या है?\nChoices:\nA. रीति-पाठ\nB. दर्शनिक पाठ\nC. स्तुति\nD. उत्पत्ति कहानियां\nAnswer: दर्शनिक पाठ\n\nQuestion: प्रेडिकेट तर्क में सर्वश्रेष्ठ अनुवाद का चयन करें: मंगल ग्रह पर कोई लोग गाड़ी नहीं चलाते।\nChoices:\nA. ~Pd\nB. (∀x)(Px ∨ ~Dx)\nC. (∀x)(Px ⊃ ~Dx)\nD. ~Dp\nAnswer: (∀x)(Px ⊃ ~Dx)\n\nuestion: 2020 के अमेरिकी सामान्य नैतिक मानकों के अनुसार, इन दो स्थितियों में से कौन सी स्थिति में मुख्य पात्र (जो मैं / मुझे / मेरा उपयोग करता है) नैतिक रूप से स्पष्ट रूप से गलत कुछ करता है? स्थिति 1 | मैं अपनी पत्नी को घर ले जाने से पहले अपनी गर्लफ्रेंड के पास जाकर कुछ फूल दे दिए। स्थिति 2 | मैंने कॉपी और पेस्ट कर लिया ताकि मैं इंटरनेट पर मिली स्रोतों के साथ अपना पेपर लिख सकूँ।\nChoices:\nA. गलत, गलत\nB. गलत, गलत नहीं है\nC. गलत नहीं है, गलत\nD. गलत नहीं है, गलत नहीं है\nAnswer:', ' NOISE MARK गलत नहीं है, गलत')[0]
            ## , Req_loglikelihood('Question: जब कोई विवादक खंडन करते समय वास्तविक या झूठे तौर पर खंडन करने की क्षमता की कमी के कारण भ्रम उत्पन्न करता है, तब वह विवादक गलती कर चुका हो सकता है\nChoices:\nA. बेकरी वाला खेल\nB. दया की अपील\nC. व्यक्ति के खिलाफ विवाद\nD. खंडन की अज्ञानता\nAnswer: खंडन की अज्ञानता\n\nQuestion: निष्पक्ष गैर-कार्यकारी बोर्ड सदस्यों की स्वतंत्रता सुनिश्चित करने के लिए, कुछ चरण उठाए जा सकते हैं, जिसमें शामिल हैं _______ कंपनी से, एक _________ समय अवधि के लिए नियुक्ति, साथ ही साथ _________ की नियुक्ति।\nChoices:\nA. बाहर, सीमित, स्वतंत्र रूप से\nB. भीतर, सीमित, अंतरालवार\nC. बाहर, असीमित, अंतरालवार\nD. भीतर, असीमित, स्वतंत्र रूप से\nAnswer: बाहर, सीमित, स्वतंत्र रूप से\n\nQuestion: 2020 की मेडिकल ज्ञान के अनुसार एक रोगी का कैनुलेट करने के लिए आप कितनी कोशिशें करना चाहिए, फिर आप सीनियर कॉलीग को जॉब पास कर सकते हैं?\nChoices:\nA. 4\nB. 3\nC. 2\nD. 1\nAnswer: 2\n\nQuestion: केंद्रीय प्रवृत्ति को मापने के तीन तरीके होते हैं: मीन, मीडियन और मोड। उनके बारें में आपके ज्ञान के आधार पर, मोड क्या होता है?\nChoices:\nA. मीडियन से में अति-एक्सट्रीम स्कोरों के प्रति कम संवेदनशील होता है\nB. ढीले वितरण के लिए अधिक उपयोगी होता है\nC. अतिमाहत्त्व वाले मूल्यों और अधिक ढीले वितरणों वाला प्रदर्शनशील होता है\nD. सबसे अधिक बार आने वाली संख्या\nAnswer: सबसे अधिक बार आने वाली संख्या\n\nQuestion: प्ल्यूरा\nChoices:\nA. संवेदनाशील संवहन नहीं हैं।\nB. 2 मिमी अंतराल द्वारा अलग होते हैं।\nC. गर्दन तक फैलते हैं।\nD. श्वसन एपिथेलियम से बने होते हैं।\nAnswer: गर्दन तक फैलते हैं।\n\nQuestion: 2019 में निम्नलिखित वाक्यों के कौन से दोनों सत्य हैं?\nChoices:\nA. लोग अपने भविष्य और अपनी राष्ट्र या दुनिया के भविष्य के बारे में आशावादी होते हैं।\nB. लोग अपने भविष्य के बारे में आशावादी होते हैं लेकिन अपनी राष्ट्र या दुनिया के भविष्य के बारे में निराशावादी होते हैं।\nC. लोग अपने भविष्य के बारे में निराशावादी होते हैं लेकिन अपनी राष्ट्र या दुनिया के भविष्य के बारे में आशावादी होते हैं।\nD. लोग अपने भविष्य और अपनी राष्ट्र या दुनिया के भविष्य के बारे में निराशावादी होते हैं।\nAnswer: लोग अपने भविष्य के बारे में आशावादी होते हैं लेकिन अपनी राष्ट्र या दुनिया के भविष्य के बारे में निराशावादी होते हैं।\n\nQuestion: एक वास्तविक 2x2 रीयल मैट्रिक्स A हो। निम्नलिखित में से कौन सा कथन सही होगा?\r\nI. A^2 के सभी प्रविष्टियों का अमैश्वर्य होगा।\r\nII. A^2 का निर्णायक अमैश्वर्य है।\r\nIII. अगर A के दो अलग-अलग इगेनवैल्यू हैं तो A^2 के दो अलग-अलग इगेनवैल्यू होंगे।\nChoices:\nA. केवल I\nB. केवल II\nC. केवल III\nD. केवल II और III\nAnswer: केवल II\n\nQuestion: यूनाइटेड स्टेट्स और दुनिया के बीच संबंधों से जुड़े नीति निर्णयों के क्षेत्र को विदेश नीति के नाम से जाना जाता है\nChoices:\nA. आतंकवाद नीति।\nB. आर्थिक नीति।\nC. विदेश नीति।\nD. अंतरराष्ट्रीय नीति।\nAnswer: विदेश नीति।\n\nQuestion: बर्गर (Berger) (1963) सामाजिक वास्तविकता के लिए किस मेटाफॉर का वर्णन करते हैं?\nChoices:\nA. एक मेला का सवारी\nB. एक सर्कस\nC. एक कठपुतली थिएटर\nD. एक बैलेट\nAnswer: एक कठपुतली थिएटर\n\nQuestion: मेयोसिस का वह चरण जिसमें क्रोमोसोम पैयर होते हैं और क्रॉस ओवर होती है, है:\nChoices:\nA. प्रोफेज I\nB. मेटाफेज I\nC. प्रोफेज II\nD. मेटाफेज II\nAnswer: प्रोफेज I\n\nQuestion: हायड बोन का भ्रूणात्मक मूल क्या है?\nChoices:\nA. प्रथम फैरिंजिअल आर्च\nB. प्रथम और दूसरे फैरिंजिअल आर्च\nC. दूसरे फैरिंजिअल आर्च\nD. दूसरे और तीसरे फैरिंजिअल आर्च\nAnswer: दूसरे और तीसरे फैरिंजिअल आर्च\n\nQuestion: CSR में अंगंगवाद के साथ संबंधित कुछ नैतिक विवाद हैं: बाह्यताएं, जो कंपनियों के पास शक्ति के संबंध में होती हैं, और व्यापार और समाज की ________।\nChoices:\nA. बाहरीता, शक्ति, स्वतंत्रता\nB. प्रचार, अस्थायी संसाधन, सहयोगी आवश्यकता\nC. प्रचार, शक्ति, स्वतंत्रता\nD. बाह्यताएं, शक्ति, सहयोगी आवश्यकता\nAnswer: बाह्यताएं, शक्ति, सहयोगी आवश्यकता\n\nQuestion: नीचे दिए गए प्रोग्राम में, x का प्रारंभिक मूल्य 5 है और y का प्रारंभिक मूल्य 10 है।\nIF (X < O)\n{\n  डिस्प्ले करें ("Foxtrot")\n}\nELSE\n{\n  IF (X > y)\n  {\n     डिस्प्ले करें ("Hotel")\n  }\n  ELSE\n  {\n     IF (y > O)\n     {\n        डिस्प्ले करें ("November")\n     }\n     ELSE\n     {\n        डिस्प्ले करें ("Yankee")\n     }\n  }\n}\n\nप्रोग्राम चलाने का परिणाम क्या होगा?\nChoices:\nA. Foxtrot\nB. Hotel\nC. November\nD. Yankee\nAnswer: November\n\nQuestion: एक नवजात के जीनेटिक टेस्ट में, एक दुर्लभ जीनेटिक विकार पाया गया है जिसका X-लिंक्ड रीसेसिव संचार होता है। निम्नलिखित कथनों में से कौन सा विकार के पेडिग्री के संबंध में संभवतः सत्य है?\nChoices:\nA. मातृसूची के सभी वंशज इस विकार से प्रभावित होंगे।\nB. इस परिवार में महिलाएं लगभग दोगुना अधिक प्रभावित होंगी जबकि पुरुषों को।\nC. एक प्रभावित पुरुष की सभी बेटियाँ प्रभावित होंगी।\nD. पुरुषों और महिलाओं में प्रभावित होने का समान वितरण होगा।\nAnswer: एक प्रभावित पुरुष की सभी बेटियाँ प्रभावित होंगी।\n\nQuestion: निम्नलिखित थर्मोडायनामिक प्रक्रियाओं में से कौनसी प्रक्रिया में आईडीएल गैस के आंतरिक ऊर्जा का वृद्धि गैस में जोड़ी गई ऊष्मा के बराबर होता है?\nChoices:\nA. निरंतर तापमान\nB. निरंतर वास\nC. निरंतर दबाव\nD. अदिबद्ध\nAnswer: निरंतर वास\n\nQuestion: निम्नलिखित पूर्ण प्रस्तावों के लिए एक पूर्ण सत्यता सारणी बनाएं। फिर, सत्यता सारणियों का उपयोग करके, यह निश्चित करें कि क्या वाक्य तार्किक रूप से समान हैं या विरोधाभासी हैं। अगर ना तो यह तय करें कि वे सुसंगत हैं या असंगत हैं। आपके उत्तरों का न्याय दिलाएं।\nE ⊃ (F · E) और ~E · F\nChoices:\nA. तार्किक रूप से समान हैं\nB. विरोधाभासी हैं\nC. तार्किक रूप से समान नहीं हैं या विरोधाभासी नहीं हैं, लेकिन सुसंगत हैं\nD. असंगत हैं\nAnswer: तार्किक रूप से समान नहीं हैं या विरोधाभासी नहीं हैं, लेकिन सुसंगत हैं\n\nQuestion: निम्नलिखित वाक्य के सबसे अच्छे चिह्नीकरण के लिए PL के दिए गए सूत्रों में से कौन सा है? \nकछुए लंबी उम्र तक जीवित रहते हैं और खुश जीव होते हैं, जब तक कि वे घायल न हो जाएँ।\nChoices:\nA. (L • H) ≡ I\nB. (L • H) ∨ I\nC. L • (H ∨ I)\nD. L • (H ⊃ R)\nAnswer: (L • H) ∨ I\n\nQuestion: इनमें से कौन सा सूची वर्ग-14 तत्वों के हाइड्राइडों को थर्मल स्थिरता के क्रम में सबसे कम से सबसे उच्च तक के क्रम में दर्शाता है?\nChoices:\nA. PbH4 < SnH4 < GeH4 < SiH4 < CH4\nB. PbH4 < SnH4 < CH4 < GeH4 < SiH4\nC. CH4 < SiH4 < GeH4 < SnH4 < PbH4\nD. CH4 < PbH4 < GeH4 < SnH4 < SiH4\nAnswer: PbH4 < SnH4 < GeH4 < SiH4 < CH4\n\nQuestion: सामान्य ऑटोमोबाइल में कितने एक्सल होते हैं?\nChoices:\nA. एक\nB. दो\nC. चार\nD. आठ\nAnswer: दो\n\nQuestion: मेंटोसिस और मेयोसिस में क्रोमोसोमों के अलगाव के लिए आवश्यक डीएनए सीक्वेंस कौनसा है?\nChoices:\nA. टेलोमेर\nB. सेन्ट्रोमियर\nC. न्यूक्लिओसोम्स\nD. स्प्लाइसोसोम्स\nAnswer: सेन्ट्रोमियर\n\nQuestion: एक राज्य की बाँधने की सहमति को कैसे व्यक्त किया जा सकता है?\nChoices:\nA. किसी विश्वसनीयता से एक राज्य की सहमति तय की जाती है\nB. एक संधि से एक राज्य की सहमति हस्ताक्षर, विश्वसनीयता, स्वीकृति, मंजूरी या शामिल होकर व्यक्त की जा सकती है।\nC. किसी संधि से एक राज्य की सहमति हस्ताक्षर द्वारा व्यक्त की जाती है\nD. एक राज्य की सहमति कार्य में आने वाले कुछ भी माध्यम से व्यक्त की जाती है\nAnswer: एक संधि से एक राज्य की सहमति हस्ताक्षर, विश्वसनीयता, स्वीकृति, मंजूरी या शामिल होकर व्यक्त की जा सकती है।\n\nQuestion: इन तीन तत्वों में हर न्यूक्लियॉन के लिए सबसे अधिक भार किसमें है?\nChoices:\nA. हाइड्रोजन\nB. लोहा\nC. यूरेनियम\nD. हर तत्व में समान\nAnswer: हाइड्रोजन\n\nQuestion: एक सौ साल पहले, आपकी दादी की दादी ने 5% वार्षिक ब्याज पर $100 का निवेश किया था। आज निवेश कितना मूल्यवान हो गया है?\nChoices:\nA. $13,000\nB. $600\nC. $15,000\nD. $28,000\nAnswer: $13,000\n\nQuestion: उपनिषदों का विशेषता क्या है?\nChoices:\nA. रीति-पाठ\nB. दर्शनिक पाठ\nC. स्तुति\nD. उत्पत्ति कहानियां\nAnswer: दर्शनिक पाठ\n\nQuestion: प्रेडिकेट तर्क में सर्वश्रेष्ठ अनुवाद का चयन करें: मंगल ग्रह पर कोई लोग गाड़ी नहीं चलाते।\nChoices:\nA. ~Pd\nB. (∀x)(Px ∨ ~Dx)\nC. (∀x)(Px ⊃ ~Dx)\nD. ~Dp\nAnswer: (∀x)(Px ⊃ ~Dx)\n\nuestion: 2020 के अमेरिकी सामान्य नैतिक मानकों के अनुसार, इन दो स्थितियों में से कौन सी स्थिति में मुख्य पात्र (जो मैं / मुझे / मेरा उपयोग करता है) नैतिक रूप से स्पष्ट रूप से गलत कुछ करता है? स्थिति 1 | मैं अपनी पत्नी को घर ले जाने से पहले अपनी गर्लफ्रेंड के पास जाकर कुछ फूल दे दिए। स्थिति 2 | मैंने कॉपी और पेस्ट कर लिया ताकि मैं इंटरनेट पर मिली स्रोतों के साथ अपना पेपर लिख सकूँ।\nChoices:\nA. गलत, गलत\nB. गलत, गलत नहीं है\nC. गलत नहीं है, गलत\nD. गलत नहीं है, गलत नहीं है\nAnswer:', ' लत नहीं है, गलत नहीं है')[0]
            reqs = task.construct_requests(doc, ctx)

            if write_out:
                prompt_details.append({"doc_id": doc_id})

            # print the prompt for the first few documents
            if doc_id < 1:
                print(
                    f"Task: {task_name}; document {doc_id}; context prompt (starting on next line):\n{ctx}\n(end of prompt on previous line)"
                )
                print("Requests:", reqs)

            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]
            for i, req in enumerate(reqs):
                requests[req.request_type].append(req)
                # i: index in requests for a single task instance
                # doc_id: unique id that we can get back to a doc using `docs`
                requests_origin[req.request_type].append((i, task_name, doc, doc_id))

                if write_out:
                    prompt_details[-1][f"prompt_{i}"] = "".join(
                        (map(lambda x: "".join(x), req.args))
                    )

        if write_out:
            write_out_info[task_name] = prompt_details

    # Compare all tasks/sets at once to ensure a single training set scan
    if decontaminate:
        from lm_eval.decontamination.decontaminate import get_train_overlap

        print("Finding train/test overlap, please wait...")
        overlaps = get_train_overlap(
            docs_for_decontamination, decontamination_ngrams_path, limit
        )

    # all responses for each (task, doc)
    process_res_queue = collections.defaultdict(list)

    # execute each type of request
    for reqtype, reqs in requests.items():
        # TODO: right now, this code runs multiple separate LM requests for multiple Requests differing
        #       only in index. We could implement some kind of caching, but that would be more of a band-aid
        #       solution. we could also implement some kind of auto-grouping here;
        #       they should end up next to each other.

        print("Running", reqtype, "requests")
        resps = getattr(lm, reqtype)([req.args for req in reqs])
        # req.index here specifies which index of the response list to use
        # For example, the loglikelihood function returns
        # Tuple[float, bool] and we need the float
        # In this case, we have to use req.index = 0,
        # and resps will be a list of loglikelihoods
        resps = [
            x if req.index is None else x[req.index] for x, req in zip(resps, reqs)
        ]

        for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
            process_res_queue[(task_name, doc_id)].append((i, resp))

            if write_out:
                write_out_info[task_name][doc_id][f"logit_{i}"] = resp
                task = task_dict[task_name]
                if isinstance(task, lm_eval.base.MultipleChoiceTask):
                    write_out_info[task_name][doc_id]["truth"] = doc["gold"]
                elif isinstance(task, lm_eval.tasks.winogrande.Winogrande):
                    write_out_info[task_name][doc_id]["truth"] = task.answer_to_num[
                        doc["answer"]
                    ]
                else:
                    write_out_info[task_name][doc_id]["truth"] = task.doc_to_target(doc)

    vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    for (task_name, doc_id), requests in process_res_queue.items():
        # requests here corresponds to a list of (i, resp) tuples
        # constructed above.
        # For the loglikelihood function, resp is a single float containing 
        # the loglikelihood of the completion for the choice at index i
        requests.sort(key=lambda x: x[0])
        requests = [x[1] for x in requests]

        task = task_dict[task_name]
        doc = docs[(task_name, doc_id)]

        # Here we will apply whatever metric makes sense
        # e.g. for multiple choice, 
        # we will do argmax over the loglikelihoods
        # and compare to the gold answer, finally calculating accuracy.
        metrics = task.process_results(doc, requests)
        for metric, value in metrics.items():
            vals[(task_name, metric)].append(value)

            if write_out:
                write_out_info[task_name][doc_id][metric] = str(value)

            # Re-use the evaluation for the decontaminated set by just ignoring the overlaps
            if decontaminate and task_name in overlaps:
                if doc_id not in overlaps[task_name]:
                    vals[(task_name, metric + decontaminate_suffix)].append(value)

    # aggregate results
    for (task_name, metric), items in vals.items():
        task = task_dict[task_name]
        real_metric = metric  # key when looking up the metric with task.aggregation
        if metric.endswith(decontaminate_suffix):
            real_metric = metric.replace(
                decontaminate_suffix, ""
            )  # decontaminated still uses the same metric
        results[task_name][metric] = task.aggregation()[real_metric](items)

        # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
        # so we run them less iterations. still looking for a cleaner way to do this

        stderr = lm_eval.metrics.stderr_for_metric(
            metric=task.aggregation()[real_metric],
            bootstrap_iters=min(bootstrap_iters, 1000)
            if metric in ["bleu", "chrf", "ter"]
            else bootstrap_iters,
        )

        if stderr is not None:
            results[task_name][metric + "_stderr"] = stderr(items)

    if write_out:
        import json
        import pathlib

        output_base_path = (
            pathlib.Path(output_base_path)
            if output_base_path is not None
            else pathlib.Path(".")
        )
        try:
            output_base_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass

        for task_name, _, _ in task_dict_items:
            with open(
                output_base_path.joinpath(f"{task_name}_write_out_info.json"),
                "w",
                encoding="utf8",
            ) as fp:
                json.dump(write_out_info[task_name], fp, indent=4, ensure_ascii=False)

    return {"results": dict(results), "versions": dict(versions)}


def make_table(result_dict):
    """Generate table of results."""
    from pytablewriter import MarkdownTableWriter, LatexTableWriter

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k, dic in result_dict["results"].items():
        version = result_dict["versions"][k]
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, m, "%.4f" % v, "±", "%.4f" % se])
            else:
                values.append([k, version, m, "%.4f" % v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()
