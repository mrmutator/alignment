import argparse
import time
import os
import subprocess

def get_time():
    return str(int(time.time()))

def get_params(args):
    params = dict()
    params['dir'] = os.path.abspath(args.dir)
    params['snt'] = os.path.abspath(args.snt)
    params['f_raw'] = os.path.abspath(args.f_raw)
    params['gold_file'] = os.path.abspath(args.gold_file)
    params['job_template_dir'] = os.path.dirname(os.path.realpath(__file__))
    params['script_dir'] = os.path.abspath(os.path.join(params['job_template_dir'], '../..'))
    return params



def generate_job(**params):
    params["filter_gold"] = ""
    if not params["gold_file"]:
        params["filter_gold"] = "#"
    with open(params['job_template_dir'] + "/template_hmt_parse_job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['dir'] + "/parse_job.sh", "w") as outfile:
        outfile.write(job_file)

def send_jobs(**params):
    log_file = open("parse_job.log", "w")

    #prepare data
    job_dir = params['dir']
    job_path = job_dir + "/parse_job.sh"
    proc_prepare = subprocess.Popen(['qsub', job_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=job_dir)
    stdout, stderr = proc_prepare.communicate()
    if stderr:
        raise Exception("Failed sending parse_job: " + stderr)
    prep_job_id = stdout.strip().split(".")[0]
    log_file.write(job_path + ": " + prep_job_id + "\n")
    log_file.write("Jobs sent successfully.\n")
    log_file.close()


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-dir", required=True)
arg_parser.add_argument("-snt", required=True)
arg_parser.add_argument("-f_raw", required=True)
arg_parser.add_argument("-gold_file", required=False, default="")
arg_parser.add_argument('-no_sub', dest='no_sub', action='store_true', required=False)
arg_parser.set_defaults(no_sub=False)


args = arg_parser.parse_args()
params = get_params(args)


generate_job(**params)

if not args.no_sub:
    send_jobs(**params)
    print "Jobs sent."
else:
    print "Jobs prepared, but not sent."
