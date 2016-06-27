import argparse
import time
import os
import subprocess
import numpy as np

def get_file_length(f):
    c = 0
    with open(f, "r") as infile:
        for _ in infile:
            c += 1
    return c

def check_num_nodes_group_size(params):
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
    check_list = ["psnt", "ibm1_table_path", "job_template_dir", "script_dir", "feature_extraction"]
    for c in check_list:
        if not os.path.exists(params[c]):
            raise Exception("Path does not exist for parameter %s: <%s>" % (c, params[c]))


def get_time():
    return str(int(time.time()))

def get_params(args):
    params = dict()
    params['pr'] = "%"
    params['dir'] = os.path.abspath(args.dir)
    params['job_name'] = args.job_name
    params['num_iterations'] = args.num_iterations
    params['psnt'] = os.path.abspath(args.psnt)
    params['feature_extraction'] = os.path.abspath(args.feature_extraction)
    params['ibm1_table_path'] = os.path.abspath(args.ibm1_table)
    params['group_size'] = args.group_size
    params['group_size_feature_extraction'] = args.group_size_feature_extraction
    params['job_template_dir'] = os.path.dirname(os.path.realpath(__file__))
    params['script_dir'] = os.path.abspath(os.path.join(params['job_template_dir'], '../..'))
    params['num_workers'] = args.num_workers
    params['buffer_size_estep'] = args.buffer_size_estep
    params['buffer_size_mstep'] = args.buffer_size_mstep
    params['p_0'] = args.p_0
    params['kappa'] = args.kappa
    params['no_prepare'] = args.no_prepare

    params['num_nodes'] = args.num_nodes
    params["hmm"] = ""
    if args.hmm:
        params["hmm"] = "-hmm"
    params["align_parts"] = 0
    params["align_limits"] = []

    if args.align_limit == -1:
        # align all
        params["align_parts"] = params["num_nodes"]
        params["align_limits"] = [0 for _ in xrange(params["num_nodes"])]
    elif args.align_limit == 0:
        pass
    else:
        params["align_parts"] = int(np.ceil(args.align_limit / float(args.group_size)))
        params["align_limits"] = [0 for _ in xrange(params["align_parts"])]
        rest = args.align_limit % args.group_size
        params["align_limits"][params["align_parts"]-1] = rest

    params['PBS_time_prepare_job'] = args.PBS_time_prepare_job
    params['PBS_time_worker_job'] = args.PBS_time_worker_job
    params['PBS_time_update_job'] = args.PBS_time_update_job
    params['PBS_time_evaluate_job'] = args.PBS_time_evaluate_job

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
    with open(params['job_template_dir'] + "/template_hmt_prepare_job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['job_dir'] + "/prepare_job.sh", "w") as outfile:
        outfile.write(job_file)

def generate_iteration_jobs(**params):
    params['it_dir'] = params["dir"] + "/it" + params["it_number"]
    params['prev_it_dir'] = params["dir"] + "/it" + str(int(params["it_number"]) -1)
    params['job_dir'] = params["dir"] + "/jobs" + params["it_number"]
    with open(params['job_template_dir'] + "/template_hmt_worker_job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['job_dir'] + "/worker_job_it" + params["it_number"] + ".sh", "w") as outfile:
        outfile.write(job_file)

    with open(params['job_template_dir'] + "/template_hmt_update_job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['job_dir'] + "/update_job_it" + params["it_number"] + ".sh", "w") as outfile:
        outfile.write(job_file)

    for part in xrange(1, params["align_parts"]+1):
        params["part"] = part
        params["vit_limit"] = ""
        if params["align_limits"][part-1] > 0:
            params["vit_limit"] = "-limit " + str(params["align_limits"][part-1])
        with open(params['job_template_dir'] + "/template_hmt_evaluate_job.txt", "r") as infile:
            template = infile.read()
            job_file = template % params

        with open(params['job_dir'] + "/evaluate_job_it" + params["it_number"] + "." + str(part) + ".sh", "w") as outfile:
            outfile.write(job_file)


def send_jobs(**params):
    log_file = open(params["job_name"] + ".log", "w")
    last_job_id = None
    #prepare data
    if not params["no_prepare"]:
        job_dir = params['dir'] + "/jobs0"
        job_path = job_dir + "/prepare_job.sh"
        proc_prepare = subprocess.Popen(['qsub', job_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=job_dir)
        stdout, stderr = proc_prepare.communicate()
        if stderr:
            raise Exception("Failed sending prepare_job: " + stderr)
        prep_job_id = stdout.strip().split(".")[0]
        log_file.write(job_path + ": " + prep_job_id + "\n")

        last_job_id = prep_job_id
    # iteration jobs
    for i in xrange(1, params["num_iterations"]+1):
        depend_string = ""
        if last_job_id:
            depend_string =  "-Wdepend=afterok:"+last_job_id
        job_dir = params['dir'] + "/jobs" + str(i)
        # workers
        job_path = job_dir + "/worker_job_it" +str(i) + ".sh"
        proc_prepare = subprocess.Popen(['qsub', depend_string, "-t", "1-"+str(params["num_nodes"]),
                                         job_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=job_dir)
        stdout, stderr = proc_prepare.communicate()
        if stderr:
            raise Exception("Failed sending worker job it" + str(i) + " : " + stderr)
        worker_array_job_id = stdout.strip().split(".")[0]
        log_file.write(job_path + ": " + worker_array_job_id + "\n")

        #update job
        job_path = job_dir + "/update_job_it" +str(i) + ".sh"
        proc_prepare = subprocess.Popen(['qsub', "-Wdepend=afterokarray:"+worker_array_job_id, job_path],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=job_dir)
        stdout, stderr = proc_prepare.communicate()
        if stderr:
            raise Exception("Failed sending update job it" + str(i) + " : " + stderr)
        update_job_id = stdout.strip().split(".")[0]
        log_file.write(job_path + ": " + update_job_id + "\n")
        last_job_id = update_job_id

        # eval job
        for part in xrange(1, params["align_parts"]+1):
            job_path = job_dir + "/evaluate_job_it" +str(i) + "." + str(part) + ".sh"
            proc_prepare = subprocess.Popen(['qsub', "-Wdepend=afterok:"+update_job_id, job_path],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=job_dir)
            stdout, stderr = proc_prepare.communicate()
            if stderr:
                raise Exception("Failed sending evaluate job it" + str(i) + "." + str(part) +  " : " + stderr)
            eval_job_id = stdout.strip().split(".")[0]
            log_file.write(job_path + ": " + eval_job_id + "\n")

    log_file.write("Jobs sent successfully.\n")
    log_file.close()


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-dir", required=True)
arg_parser.add_argument("-job_name", required=False, default=get_time())
arg_parser.add_argument("-feature_extraction", required=True)

arg_parser.add_argument("-psnt", required=True)
arg_parser.add_argument("-ibm1_table", required=True, default="")

arg_parser.add_argument("-num_iterations", required=True, type=int)

arg_parser.add_argument("-p_0", required=False, default=0.2, type=float)
arg_parser.add_argument("-kappa", required=False, default=0.001, type=float)

arg_parser.add_argument('-hmm', dest="hmm", action="store_true", required=False)


arg_parser.add_argument("-group_size", required=True, type=int)
arg_parser.add_argument("-group_size_feature_extraction", required=False, type=int, default=-1)
arg_parser.add_argument("-num_nodes", required=True, type=int)

arg_parser.add_argument("-num_workers", required=False, default=16, type=int)
arg_parser.add_argument("-buffer_size_estep", required=False, default=200, type=int)
arg_parser.add_argument("-buffer_size_mstep", required=False, default=20, type=int)

arg_parser.add_argument("-align_limit", required=False, default=-1, type=int)


arg_parser.add_argument('-no_sub', dest='no_sub', action='store_true', required=False)
arg_parser.add_argument('-no_prepare', dest='no_prepare', action='store_true', required=False)
arg_parser.set_defaults(no_sub=False)
arg_parser.set_defaults(no_prepare=False)

arg_parser.add_argument('-ignore_checks', dest='ignore_checks', action='store_true', required=False)
arg_parser.set_defaults(ignore_checks=False)

arg_parser.add_argument("-PBS_time_prepare_job", required=False, default="01:30:00", type=str)
arg_parser.add_argument("-PBS_time_worker_job", required=False, default="00:20:00", type=str)
arg_parser.add_argument("-PBS_time_update_job", required=False, default="00:20:00", type=str)
arg_parser.add_argument("-PBS_time_evaluate_job", required=False, default="01:00:00", type=str)
args = arg_parser.parse_args()
params = get_params(args)

check_paths(params)
if not args.ignore_checks:
    check_num_nodes_group_size(params)


# make directories
make_directories(params['dir'], params['num_iterations'])

generate_prepare_job(**params)

for i in xrange(1, params["num_iterations"]+1):
    params["it_number"] = str(i)
    generate_iteration_jobs(**params)

if not args.no_sub:
    send_jobs(**params)
    print "Jobs sent."
else:
    print "Jobs prepared, but not sent."
