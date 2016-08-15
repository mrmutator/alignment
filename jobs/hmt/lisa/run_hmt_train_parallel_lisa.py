import argparse
import os
import shutil

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
    assert params["num_iterations"] > 0
    params['job_template_dir'] = os.path.dirname(os.path.realpath(__file__))
    params['script_dir'] = os.path.abspath(os.path.join(params['job_template_dir'], '../../../'))
    params['num_workers'] = int(config_dict["num_workers"])
    params['p_0'] = float(config_dict["p_0"])
    params['alpha'] = float(config_dict["alpha"])
    params['num_nodes'] = int(config_dict["num_nodes"])
    assert params['num_nodes'] > 0

    params["align_limit"] = int(config_dict["align_limit"])

    if params["align_limit"] == 0:
        # align all
        params["align_parts"] = params["num_nodes"]
    else:
        params["align_parts"] = 1

    params['wall_time_train'] = config_dict["wall_time_train"]
    params['wall_time_update'] = config_dict["wall_time_update"]
    params['wall_time_evaluate'] = config_dict["wall_time_evaluate"]

    return params


def make_directory(path_it):
    if not os.path.exists(path_it):
        os.makedirs(path_it)

def generate_iteration_jobs(**params):
    params['it_dir'] = params["result_dir"] + "/it" + params["it_number"]
    make_directory(params['it_dir'])
    if params["it_number"] == "1":
        params['prev_it_dir'] = params['it0_dir']
    else:
        params['prev_it_dir'] = params["result_dir"] + "/it" + str(int(params["it_number"]) -1)
    params['job_dir'] = params["result_dir"] + "/jobs" + params["it_number"]
    make_directory(params['job_dir'])
    with open(params['job_template_dir'] + "/template_hmt_worker_job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['job_dir'] + "/train_job_it" + params["it_number"] + ".job", "w") as outfile:
        outfile.write(job_file)

    with open(params['job_template_dir'] + "/template_hmt_update_job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['job_dir'] + "/update_job_it" + params["it_number"] + ".job", "w") as outfile:
        outfile.write(job_file)

    for part in xrange(1, params["align_parts"]+1):
        params["part"] = part
        with open(params['job_template_dir'] + "/template_hmt_evaluate_job.txt", "r") as infile:
            template = infile.read()
            job_file = template % params

        with open(params['job_dir'] + "/evaluate_job_it" + params["it_number"] + "." + str(part) + ".job", "w") as outfile:
            outfile.write(job_file)


def write_dependency_file(**params):
    job_log_file = os.path.abspath(os.path.join(params["result_dir"], params["job_name"] + "_iterations.jobs"))
    job_id = 0
    with open(job_log_file, "w") as outfile:
        if params["dep_job"]:
            outfile.write(" ".join(["-1 ", params["dep_job"], "-", "-"]) + "\n")
            outfile.write(" ".join([str(job_id), params["result_dir"] + "/jobs1/train_job_it1.job", "1-"+str(params["num_nodes"]), "-1"]) + "\n")
            job_id += 1
        else:
            outfile.write(" ".join([str(job_id), params["result_dir"] + "/jobs1/train_job_it1.job", "1-"+str(params["num_nodes"]), "-"]) + "\n")
            job_id += 1
        outfile.write(" ".join([str(job_id), params["result_dir"] + "/jobs1/update_job_it1.job", "-", str(job_id-1)]) + "\n")
        update_id = job_id
        job_id += 1
        for part in xrange(1, params["align_parts"]+1):
            outfile.write(" ".join(
                [str(job_id), params["result_dir"] + "/jobs1/evaluate_job_it1." + str(part) + ".job", "-", str(update_id)]) + "\n")
            job_id += 1

        for it in xrange(2, params["num_iterations"]+1):
            outfile.write(" ".join(
                [str(job_id), params["result_dir"] + "/jobs" + str(it) + "/train_job_it" + str(it) +".job", "1-" + str(params["num_nodes"]),
                 str(update_id)]) + "\n")
            job_id += 1
            outfile.write(
                " ".join([str(job_id), params["result_dir"] + "/jobs" + str(it) + "/update_job_it" + str(it) + ".job", "-", str(job_id - 1)]) + "\n")
            update_id = job_id
            job_id += 1
            for part in xrange(1, params["align_parts"] + 1):
                outfile.write(" ".join(
                    [str(job_id), params["result_dir"] + "/jobs" + str(it) + "/evaluate_job_it" + str(it) + "." + str(part) + ".job", "-",
                    str(update_id)]) + "\n")
                job_id += 1


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-config_file")
arg_parser.add_argument("-dep_job", default="")

args = arg_parser.parse_args()

if not args.config_file:
    template = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "train_job.cfg"))
    if not os.path.exists(os.path.abspath("./train_job.cfg")):
        shutil.copy(template, "./train_job.cfg")
else:
    params = parse_config(args.config_file)
    if args.dep_job:
        params["dep_job"] = args.dep_job
    else:
        params["dep_job"] = None

    # make directories
    make_directory(params['result_dir'])

    for i in xrange(1, params["num_iterations"] + 1):
        params["it_number"] = str(i)
        generate_iteration_jobs(**params)

    write_dependency_file(**params)


    print "Job prepared, but not sent."
