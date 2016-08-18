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

def check_num_nodes_group_size(params):
    if params["group_size"] <= 0:
        return
    psnt = params['psnt']
    raw_length = get_file_length(psnt)
    file_length = raw_length / 8
    assert raw_length % 8 == 0
    covered = params["group_size"] * params["num_nodes"]
    if file_length > covered:
        raise Exception("Current group_size / number of node configuration does not cover entire corpus file.")
    if np.ceil(float(file_length) / params["group_size"]) < params["num_nodes"]:
        raise Exception("Too many nodes specified for current configuration.")

def check_paths(params):
    check_list = ["psnt", "ibm1_table", "job_template_dir", "script_dir"]
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
    params['it0_dir'] = os.path.abspath(config_dict["it0_dir"])
    params['psnt'] = os.path.abspath(config_dict["psnt"])
    params['ibm1_table'] = os.path.abspath(config_dict["ibm1_table"])
    params['job_name'] = config_dict["job_name"]
    params['group_size'] = int(config_dict["group_size"])
    params['job_template_dir'] = os.path.dirname(os.path.realpath(__file__))
    params['script_dir'] = os.path.abspath(os.path.join(params['job_template_dir'], '../../../'))
    params['num_nodes'] = int(config_dict["num_nodes"])
    assert params['num_nodes'] > 0
    params['wall_time'] = config_dict["wall_time"]
    params["tj_cond_tok"] = "-tj_cond_tok " + config_dict["tj_cond_tok"] if config_dict["tj_cond_tok"] != "-" else ""
    params["tj_cond_head"] = "-tj_cond_head " + config_dict["tj_cond_head"] if config_dict["tj_cond_head"] != "-" else ""
    params["start_cond_tok"] = "-start_cond_tok " + config_dict["start_cond_tok"] if config_dict["start_cond_tok"] != "-" else ""
    params['num_workers'] = int(config_dict["num_workers"])
    assert config_dict["uniform_parameters"].lower() in ["true", "false"]
    params['uniform'] =  "-uniform" if config_dict["uniform_parameters"].lower() == "true" else ""

    return params


def make_it0_directories(path_it):
    if not os.path.exists(path_it):
        os.makedirs(path_it)

def generate_prepare_job(**params):
    with open(params['job_template_dir'] + "/template_hmt_prepare_job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['dir'] + "/" + params["job_name"] + "_prepare.job", "w") as outfile:
        outfile.write(job_file)

def generate_train_config(**params):
    with open(params['job_template_dir'] + "/train_job.cfg", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['dir'] + "/" + params["job_name"] + "_train.cfg", "w") as outfile:
        outfile.write(job_file)


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-config_file", required=False)

args = arg_parser.parse_args()
if not args.config_file:
    template = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "prepare_job.cfg"))
    if not os.path.exists(os.path.abspath("./prepare_job.cfg")):
        shutil.copy(template, "./prepare_job.cfg")
else:
    params = parse_config(args.config_file)
    check_paths(params)
    check_num_nodes_group_size(params)

    # make directories
    make_it0_directories(params['it0_dir'])

    generate_prepare_job(**params)
    generate_train_config(**params)
    print "Job prepared, but not sent."
