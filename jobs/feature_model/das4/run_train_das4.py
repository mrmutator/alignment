import argparse
import time
import os
import shutil

def get_time():
    return str(int(time.time()))

def check_paths(params):
    check_list = ["it0_dir"]
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
    params['it0_dir'] = os.path.abspath(config_dict["it0_dir"])
    params['job_name'] = config_dict["job_name"]
    params['num_iterations'] = int(config_dict["num_iterations"])
    params['job_template_dir'] = os.path.dirname(os.path.realpath(__file__))
    params['script_dir'] = os.path.abspath(os.path.join(params['job_template_dir'], '../../../'))
    params['num_workers'] = int(config_dict["num_workers"])
    params['p_0'] = float(config_dict["p_0"])
    params['kappa'] = float(config_dict["kappa"])
    params['num_nodes'] = int(config_dict["num_nodes"])
    assert params['num_nodes'] > 0

    params["align_limit"] = config_dict["align_limit"]

    if params["align_limit"] == 0:
        # align all
        params["align_parts"] = params["num_nodes"]
    else:
        params["align_parts"] = 1


    params['wall_time'] = config_dict["wall_time"]

    params['single_processes'] = True if config_dict["single_processes"].strip().lower() == "true" else False

    return params


def make_result_directories(path_it):
    if not os.path.exists(path_it):
        os.makedirs(path_it)

def generate_single_job(**params):
    template_file = "/template_train_job.txt"
    if params["single_processes"]:
        template_file = "/template_single_processes_job.txt"
    with open(params['job_template_dir'] + template_file, "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['dir'] + "/" + params["job_name"] + "_train.job", "w") as outfile:
        outfile.write(job_file)


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-config_file")

args = arg_parser.parse_args()

if not args.config_file:
    template = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "train_job.cfg"))
    if not os.path.exists(os.path.abspath("./train_job.cfg")):
        shutil.copy(template, "./train_job.cfg")
else:
    params = parse_config(args.config_file)

    # make directories
    make_result_directories(params['result_dir'])

    generate_single_job(**params)
    print "Job prepared, but not sent."
