import argparse
import time
import os
import subprocess

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
    params['num_nodes'] = args.num_nodes
    params["gold"] = args.gold
    params["align1"] = args.align1

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

    if params["align1"]:
        with open(params['job_template_dir'] + "/template_hmm_evaluate_job.txt", "r") as infile:
            template = infile.read()
            job_file = template % params
        if params["gold"]:
            job_file += "python %(script_dir)s/utils/evaluate.py -gold %(gold)s -test %(it_dir)s/%(job_name)s.1.aligned " \
                        ">> %(it_dir)s/eval.txt\n"  % params
            job_file += "python %(script_dir)s/utils/evaluate.py -gold %(gold)s -test %(it_dir)s/%(job_name)s.1.aligned " \
                        ">> %(it_dir)s/eval.txt -swap\n"  % params

        with open(params['job_dir'] + "/evaluate_job_it" + params["it_number"] + ".sh", "w") as outfile:
            outfile.write(job_file)


def send_jobs(**params):
    log_file = open(params["job_name"] + ".log", "w")

    #prepare data
    job_path = params['dir'] + "/jobs0/prepare_job.sh"
    proc_prepare = subprocess.Popen(['qsub', job_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc_prepare.communicate()
    if stderr:
        raise Exception("Failed sending prepare_job: " + stderr)
    prep_job_id = stdout.strip()
    log_file.write(job_path + ": " + prep_job_id + "\n")

    last_job_id = prep_job_id
    # iteration jobs
    for i in xrange(1, params["num_iterations"]+1):
        # workers
        job_path = params['dir'] + "/jobs" + str(i) + "/worker_job_it" +str(i) + ".sh"
        proc_prepare = subprocess.Popen(['qsub', "-Wdepdend:afterok:"+last_job_id, "-t", "1-"+str(params["num_nodes"]),
                                         job_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc_prepare.communicate()
        if stderr:
            raise Exception("Failed sending worker job it" + str(i) + " : " + stderr)
        worker_array_job_id = stdout.strip()
        log_file.write(job_path + ": " + worker_array_job_id + "\n")

        #update job
        job_path = params['dir'] + "/jobs" + str(i) + "/update_job_it" +str(i) + ".sh"
        proc_prepare = subprocess.Popen(['qsub', "-Wdepdend:afterokarray:"+worker_array_job_id, job_path],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc_prepare.communicate()
        if stderr:
            raise Exception("Failed sending update job it" + str(i) + " : " + stderr)
        update_job_id = stdout.strip()
        log_file.write(job_path + ": " + update_job_id + "\n")
        last_job_id = update_job_id

        # eval job
        if params["align1"]:
            job_path = params['dir'] + "/jobs" + str(i) + "/evaluate_job_it" +str(i) + ".sh"
            proc_prepare = subprocess.Popen(['qsub', "-Wdepdend:afterok:"+update_job_id, job_path],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc_prepare.communicate()
            if stderr:
                raise Exception("Failed sending evaluate job it" + str(i) + " : " + stderr)
            eval_job_id = stdout.strip()
            log_file.write(job_path + ": " + eval_job_id + "\n")

    log_file.write("Jobs sent successfully.\n")


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

arg_parser.add_argument("-group_size", required=True, type=int)
arg_parser.add_argument("-num_workers", required=False, default=16, type=int)

arg_parser.add_argument("-num_nodes", required=True, type=int)

arg_parser.add_argument("-e_test", required=False, type=int)

arg_parser.add_argument('-no_sub', dest='no_sub', action='store_true', required=False)
arg_parser.set_defaults(no_sub=False)

arg_parser.add_argument('-align1', dest="align1", action="store_true", required=False)
arg_parser.set_defaults(align1=False)

arg_parser.add_argument("-gold", required=False, default="")

args = arg_parser.parse_args()
params = get_params(args)


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
