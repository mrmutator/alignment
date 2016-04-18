import argparse
import time
import os
import subprocess


def get_time():
    return str(int(time.time()))


def get_params(args):
    params = dict()
    params['dir'] = os.path.abspath(args.dir)
    params['e_file'] = os.path.abspath(args.e)
    params['f_file'] = os.path.abspath(args.f)
    params['giza_dir'] = os.path.abspath(args.giza_dir)
    params['PBS_time_prepare_job'] = args.PBS_time_prepare_job
    params['PBS_time_parse_job'] = args.PBS_time_parse_job
    params['PBS_time_mgiza_job'] = args.PBS_time_mgiza_job
    params['gold_file'] = os.path.abspath(args.gold)

    params['snt_file'] = os.path.abspath(os.path.join(params['dir'], 'ef.snt'))
    params['e_vcb'] = os.path.abspath(os.path.join(params['dir'], 'e.vcb'))
    params['f_vcb'] = os.path.abspath(os.path.join(params['dir'], 'f.vcb'))
    params['filter_file'] = os.path.abspath(os.path.join(params['dir'], 'filter.txt'))
    params['snt_filtered'] = os.path.abspath(os.path.join(params['dir'], 'ef.snt.filtered'))
    params['psnt_file'] = os.path.abspath(os.path.join(params['dir'], 'ef.psnt'))
    params['pos_voc_file'] = os.path.abspath(os.path.join(params['dir'], 'ef.posvoc'))
    params['rel_voc_file'] = os.path.abspath(os.path.join(params['dir'], 'ef.relvoc'))
    params['gold_file_filtered'] = os.path.abspath(os.path.join(params['dir'], 'gold.filtered'))
    params['ibm1_table'] = os.path.abspath(os.path.join(params['dir'], 'ibm1.table'))
    params['e_filtered'] = os.path.abspath(os.path.join(params['dir'], 'corpus.filtered.e'))
    params['f_filtered'] = os.path.abspath(os.path.join(params['dir'], 'corpus.filtered.f'))

    params['job_dir'] = os.path.abspath(os.path.join(params["dir"], "jobs"))
    params["giza_result_dir"] = os.path.abspath(os.path.join(params["dir"], "giza"))
    params['giza_aligned'] = os.path.abspath(os.path.join(params['giza_result_dir'], 'giza.aligned'))
    params['job_template_dir'] = os.path.dirname(os.path.realpath(__file__))
    params['script_dir'] = os.path.abspath(os.path.join(params['job_template_dir'], '../..'))

    params["gold_cmt"] = ""
    if not args.gold:
        params["gold_cmt"] = "#"

    return params


def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_prepare_job(**params):
    with open(params['job_template_dir'] + "/template_prepare_dataset_job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['job_dir'] + "/prepare_dataset_job.sh", "w") as outfile:
        outfile.write(job_file)


def generate_parse_job(**params):
    with open(params['job_template_dir'] + "/template_parse_job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['job_dir'] + "/parse_job.sh", "w") as outfile:
        outfile.write(job_file)


def generate_mgiza_job(**params):
    with open(params['job_template_dir'] + "/template_mgiza_job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['job_dir'] + "/mgiza_job.sh", "w") as outfile:
        outfile.write(job_file)


def send_jobs(**params):
    log_file = open(params["job_name"] + ".log", "w")
    job_dir = params["job_dir"]

    # prepare dataset job
    job_path = os.path.abspath(os.path.join(job_dir, "prepare_dataset_job.sh"))
    proc = subprocess.Popen(['qsub', job_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=job_dir)
    stdout, stderr = proc.communicate()
    if stderr:
        raise Exception("Failed sending prepare_job: " + stderr)
    prepare_job_id = stdout.strip().split(".")[0]
    log_file.write(job_path + ": " + prepare_job_id + "\n")

    # parse job
    job_path = os.path.abspath(os.path.join(job_dir, "parse_job.sh"))
    proc = subprocess.Popen(['qsub', "-Wdepend=afterok:" + prepare_job_id, job_path], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=job_dir)
    stdout, stderr = proc.communicate()
    if stderr:
        raise Exception("Failed sending parse_job: " + stderr)
    parse_job_id = stdout.strip().split(".")[0]
    log_file.write(job_path + ": " + parse_job_id + "\n")

    # mgiza job
    job_path = os.path.abspath(os.path.join(job_dir, "mgiza_job.sh"))
    proc = subprocess.Popen(['qsub', "-Wdepend=afterok:" + parse_job_id, job_path], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=job_dir)
    stdout, stderr = proc.communicate()
    if stderr:
        raise Exception("Failed sending mgiza_job: " + stderr)
    mgiza_job_id = stdout.strip().split(".")[0]
    log_file.write(job_path + ": " + mgiza_job_id + "\n")

    log_file.write("Jobs sent successfully.\n")
    log_file.close()


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-dir", required=True)
arg_parser.add_argument("-e", required=True)
arg_parser.add_argument("-f", required=True)
arg_parser.add_argument("-gold", required=False, type=str, default="")

arg_parser.add_argument("-PBS_time_prepare_job", required=False, default="00:05:00", type=str)
arg_parser.add_argument("-PBS_time_parse_job", required=False, default="01:00:00", type=str)
arg_parser.add_argument("-PBS_time_mgiza_job", required=False, default="01:00:00", type=str)
arg_parser.add_argument("-giza_dir", required=False, default="$home/mgiza", type=str)

arg_parser.add_argument('-no_sub', dest='no_sub', action='store_true', required=False)
arg_parser.set_defaults(no_sub=False)

args = arg_parser.parse_args()
params = get_params(args)

# make directories
make_directory(params["giza_result_dir"])
make_directory(params["job_dir"])

generate_prepare_job(**params)
generate_parse_job(**params)
generate_prepare_job(**params)
generate_mgiza_job(**params)

if not args.no_sub:
    send_jobs(**params)
    print "Jobs sent."
else:
    print "Jobs prepared, but not sent."
