import argparse
import time
import os

def get_time():
    return str(int(time.time()))

def get_params(args):
    params = dict()
    params['dir'] = os.path.abspath(args.dir)
    params['job_name'] = args.job_name
    params['num_iterations'] = args.num_iterations
    params['e_path'] = os.path.abspath(args.e)
    params['f_path'] = os.path.abspath(args.f)
    params['ibm1_table_path'] = os.path.abspath(args.ibm1_table)
    params['e_vocab_path'] = os.path.abspath(args.e_vocab)
    params['f_vocab_path'] = os.path.abspath(args.f_vocab)
    params['group_size'] = args.group_size
    params['job_template_dir'] = os.path.dirname(os.path.realpath(__file__))
    params['script_dir'] = os.path.abspath(os.path.join(params['job_template_dir'], '..'))
    params['num_workers'] = args.num_workers
    params['alpha'] = args.alpha
    params['p_0'] = args.p_0

    return params



def make_directories(dir, num_iterations):
    for i in xrange(0, num_iterations + 1):
        path_it = dir + "/it" + str(i) + "/"
        path_job = dir  + "/jobs" + str(i) + "/"
        if not os.path.exists(path_it):
            os.makedirs(path_it)
        if not os.path.exists(path_job):
            os.makedirs(path_job)

def generate_prepare_job(**params):
    params['it_dir'] = params["dir"] + "/it0"
    params['job_dir'] = params["dir"] + "/jobs0"
    with open(params['job_template_dir'] + "/template_hmm_prepare_job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['job_dir'] + "/prepare_job.sh", "w") as outfile:
        outfile.write(job_file)

def generate_iteration_jobs(**params):
    params['it_dir'] = params["dir"] + "/it" + params["it_number"]
    params['job_dir'] = params["dir"] + "/jobs" + params["it_number"]
    with open(params['job_template_dir'] + "/template_hmm_worker_job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['job_dir'] + "/worker_job_it" + params["it_number"] + ".sh", "w") as outfile:
        outfile.write(job_file)

    with open(params['job_template_dir'] + "/template_hmm_update_job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['job_dir'] + "/update_job_it" + params["it_number"] + ".sh", "w") as outfile:
        outfile.write(job_file)



arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-dir", required=True)
arg_parser.add_argument("-job_name", required=False, default=get_time())
arg_parser.add_argument("-script_dir", required=True)

arg_parser.add_argument("-e", required=True)
arg_parser.add_argument("-f", required=True)

arg_parser.add_argument("-ibm1_table", required=True, default="")
arg_parser.add_argument("-e_vocab", required=True, default="")
arg_parser.add_argument("-f_vocab", required=True, default="")

arg_parser.add_argument("-num_iterations", required=True, type=int)

arg_parser.add_argument("-alpha", required=False, default=0.0, type=float)
arg_parser.add_argument("-p_0", required=False, default=0.2, type=float)

arg_parser.add_argument("-group_size", required=False, default=-1, type=int)
arg_parser.add_argument("-num_workers", required=False, default=16, type=int)

args = arg_parser.parse_args()



params = get_params(args)



# make directories
make_directories(params['dir'], params['num_iterations'])

generate_prepare_job(**params)

for i in xrange(1, params["num_iterations"]+1):
    params["it_number"] = str(i)
    generate_iteration_jobs(**params)

# To-Do
# smart way to automatically define number of nodes
