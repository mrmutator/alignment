import os
import argparse


def parse_config(config_file):
    corpus_files = {"hmm": "ef.psnt.hmm", "hmt": "ef.psnt", "mixed": "ef.psnt.mixed", "head_chain": "ef.psnt.headchain"}

    standard_params = dict()
    corpus_types = []
    experiments = []
    language_pairs = []

    with open(config_file, "r") as infile:
        for line in infile:
            if line.strip():
                if line.strip() == "<END>":
                    break
                els = line.split()
                k = els[0]
                if k == "corpus_types":
                    for c in els[1:]:
                        corpus_types.append((c, corpus_files[c]))
                elif k == "experiments":
                    for c in els[1:]:
                        experiments.append(c)
                elif k == "language_pairs":
                    for c in els[1:]:
                        language_pairs.append(c)
                else:
                    standard_params[k] = els[1]

        standard_params["uniform"] = True if standard_params["uniform"].lower() == "true" else False
        standard_params["num_workers"] = int(standard_params["num_workers"])
        standard_params["num_iterations"] = int(standard_params["num_iterations"])
        standard_params["p_0"] = float(standard_params["p_0"])
        standard_params["kappa"] = float(standard_params["kappa"])
        standard_params["lbfgs_maxiter"] = int(standard_params["lbfgs_maxiter"])
        standard_params["max_jump"] = int(standard_params["max_jump"])

    return standard_params, corpus_types, experiments, language_pairs

def get_test_set_length(gold_file):
    last = 0
    with open(gold_file, "r") as infile:
        for line in infile:
            if line.strip():
                els = line.split()
                last = int(els[0])
    return last


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_file")

    args = arg_parser.parse_args()


    STANDARD_PARAMS, CORPUS_TYPES, EXPERIMENTS, LANGUAGE_PAIRS = parse_config(args.config_file)

    jobs_file = open(os.path.join(STANDARD_PARAMS["result_dir"], "all_jobs.txt"), "w")
    job_id = 0

    this_dir = os.path.dirname(os.path.realpath(__file__))
    hmt_job_dir = os.path.abspath(os.path.join(this_dir, 'feature_model/lisa'))
    tmp = {"script_dir": os.path.abspath(os.path.join(this_dir, "../")), "pr": "%", "group_size":-1,
           "uniform": "-uniform", "num_nodes":1}
    if not STANDARD_PARAMS["uniform"]:
        tmp["uniform"] = ""

    STANDARD_PARAMS.update(tmp)

    with open(os.path.join(hmt_job_dir, "template_hmt_prepare_job.txt"), "r") as infile:
        prepare_template = infile.read()

    with open(os.path.join(hmt_job_dir, "template_single_job.txt"), "r") as infile:
        train_template = infile.read()


    for lang in LANGUAGE_PAIRS:
        data_set_dir = os.path.join(os.path.abspath(STANDARD_PARAMS["data_dir"]), lang)
        if not os.path.exists(data_set_dir):
            print "Skipping language pair: ", lang
            continue
        lang_dir = os.path.join(os.path.abspath(STANDARD_PARAMS["result_dir"]), lang)
        if not os.path.exists(lang_dir):
            os.makedirs(lang_dir)
        gold_file = os.path.join(STANDARD_PARAMS["data_dir"], lang + "/gold.filtered")
        test_set_length = get_test_set_length(gold_file)
        for corpus_label, corpus_file in CORPUS_TYPES:
            corpus_dir = os.path.join(lang_dir, corpus_label)
            if not os.path.exists(corpus_dir):
                os.makedirs(corpus_dir)

            for exp in EXPERIMENTS:
                params = {"lang": lang, "corpus": corpus_file, "job_name": exp, "align_limit": test_set_length,
                          "psnt": os.path.join(data_set_dir, corpus_file),
                          "ibm1_table": os.path.join(data_set_dir, "ibm1.table"),
                          "it0_dir": os.path.join(corpus_dir, exp)}
                params.update(STANDARD_PARAMS)
                params["wall_time"] = params["prep_wall_time"]
                prepare_file = os.path.join(params["it0_dir"], "prepare.job")
                if not os.path.exists(params["it0_dir"]):
                    os.makedirs(params["it0_dir"])
                with open(prepare_file, "w") as outfile:
                    outfile.write(prepare_template % params)

                params["result_dir"] = os.path.join(params["it0_dir"], "results")
                if not os.path.exists(params["result_dir"]):
                    os.makedirs(params["result_dir"])

                train_file = os.path.join(params["result_dir"], "train.job")
                params["wall_time"] = params["train_wall_time"]
                params["align_parts"] = 1
                with open(train_file, "w") as outfile:
                    outfile.write(train_template % params)

                jobs_file.write(" ".join(map(str, [job_id, prepare_file, "-", "-"])) + "\n")
                job_id += 1
                jobs_file.write(" ".join(map(str, [job_id, train_file, "-", job_id-1])) + "\n")
                job_id += 1

    jobs_file.close()