import argparse
import time
import os
import glob

def get_time():
    return str(int(time.time()))

def get_params(args):
    params = dict()
    params['pr'] = "%"
    params['dir'] = os.path.abspath(args.dir)
    params['result_dir'] = os.path.abspath(os.path.join(params['dir'], 'results'))
    params['it0_dir'] = os.path.abspath(args.it0_dir)
    params['job_name'] = glob.glob(params['it0_dir'] + "/*.params.*.gz")[0].split(".params.")[0].split("/")[-1]
    params['num_iterations'] = args.num_iterations
    params['job_template_dir'] = os.path.dirname(os.path.realpath(__file__))
    params['script_dir'] = os.path.abspath(os.path.join(params['job_template_dir'], '../..'))
    params['num_workers'] = args.num_workers
    params['p_0'] = args.p_0
    params['kappa'] = args.kappa
    params['num_nodes'] = len(glob.glob(params['it0_dir'] + "/*.params.*.gz"))
    assert params['num_nodes'] > 0

    params["align_limit"] = args.align_limit

    if args.align_limit == 0:
        # align all
        params["align_parts"] = params["num_nodes"]
    else:
        params["align_parts"] = 1


    params['PBS_time_single_job'] = args.PBS_time_single_job

    return params


def make_result_directories(dir):
    path_it = dir + "/results/"
    if not os.path.exists(path_it):
        os.makedirs(path_it)

def generate_single_job(**params):
    with open(params['job_template_dir'] + "/template_single_job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['dir'] + "/" + params["job_name"] + ".job", "w") as outfile:
        outfile.write(job_file)


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-dir", required=True)
arg_parser.add_argument("-it0_dir", required=True)
arg_parser.add_argument("-num_iterations", required=True, type=int)

arg_parser.add_argument("-p_0", required=False, default=0.2, type=float)
arg_parser.add_argument("-kappa", required=False, default=0.001, type=float)

arg_parser.add_argument("-num_workers", required=False, default=16, type=int)

arg_parser.add_argument("-align_limit", required=False, default=0, type=int)


arg_parser.add_argument("-PBS_time_single_job", required=False, default="10:00:00", type=str)
args = arg_parser.parse_args()
params = get_params(args)


# make directories
make_result_directories(params['dir'])

generate_single_job(**params)
print "Job prepared, but not sent."
