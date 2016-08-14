import argparse
import os
import numpy as np
import shutil

def get_file_length(f):
    c = 0
    with open(f, "r") as infile:
        for _ in infile:
            c += 1
    return c

def check_paths(params):
    check_list = ["psnt", "ibm1_table", "e_voc", "f_voc", "script_dir", "gold_file"]
    for c in check_list:
        if not os.path.exists(params[c]):
            raise Exception("Path does not exist for parameter %s: <%s>" % (c, params[c]))


def parse_config(config_file):

    config_dict = dict()

    with open(config_file, "r") as infile:
        for line in infile:
            if line.strip():
                if line.strip() == "<END>":
                    break
                k, v = line.split()
                config_dict[k] = v
    params = dict()
    params['pr'] = "%"
    params['dir'] = os.path.abspath("./")
    params['result_dir'] = os.path.abspath(config_dict["result_dir"])
    params['psnt'] = os.path.abspath(config_dict["psnt"])
    params['job_name'] = config_dict["job_name"]
    params['training_size'] = int(config_dict["training_size"])
    params['job_template_dir'] = os.path.dirname(os.path.realpath(__file__))
    params['script_dir'] = os.path.abspath(os.path.join(params['job_template_dir'], '../../../'))
    params['num_workers'] = int(config_dict["num_workers"])
    params['sigma'] = float(config_dict["num_workers"])
    assert params['num_workers'] > 0
    params['wall_time'] = config_dict["wall_time"]
    params['gold_order'] = config_dict["gold_order"]
    assert params['gold_order'] in ['ef', 'fe']

    params['corpus_dir'] = os.path.dirname(os.path.realpath(params['psnt']))
    params['ibm1_table'] = os.path.abspath(os.path.join(params['corpus_dir'], 'ibm1.table'))
    params['e_voc'] = os.path.abspath(os.path.join(params['corpus_dir'], 'e.vcb'))
    params['f_voc'] = os.path.abspath(os.path.join(params['corpus_dir'], 'f.vcb'))
    params['gold_file'] = os.path.abspath(os.path.join(params['corpus_dir'], 'gold.filtered'))


    return params


def make_it0_directories(path_it):
    if not os.path.exists(path_it):
        os.makedirs(path_it)

def generate_crf_job(**params):
    with open(params['job_template_dir'] + "/template_crf_job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['result_dir'] + "/" + params["job_name"] + "_crf.job", "w") as outfile:
        outfile.write(job_file)


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-config_file", required=False)

args = arg_parser.parse_args()
if not args.config_file:
    template = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "crf_job.cfg"))
    if not os.path.exists(os.path.abspath("./crf_job.cfg")):
        shutil.copy(template, "./crf_job.cfg")
else:
    params = parse_config(args.config_file)
    check_paths(params)

    # make directories
    make_it0_directories(params['result_dir'])

    generate_crf_job(**params)
    print "Job prepared, but not sent."
