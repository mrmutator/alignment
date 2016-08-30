import argparse
import os
import glob



def parse_config(args):

    params = dict()
    params['dir'] = os.path.abspath(args.dir)
    params['result_dir'] = params["dir"]
    params['convoc_params'] = os.path.join(params["dir"] + "convoc.1.params")
    params['params'] = os.path.join(params["dir"] + "params.1")
    params['corpus_gz'] = os.path.abspath(glob.glob(params["dir"] + "/../*.corpus.1.gz")[0])
    params['corpus_gz'] = os.path.abspath(glob.glob(params["dir"] + "/../*.order.gz")[0])
    params['job_template_dir'] = os.path.dirname(os.path.realpath(__file__))
    params['script_dir'] = os.path.abspath(os.path.join(params['job_template_dir'], '../../../'))
    params['num_workers'] = 16
    params['alpha'] = float(args.alpha)
    params['p_0'] = float(args.p_0)

    params['wall_time'] = args.wall_time

    return params


def generate_single_job(**params):
    with open(params['job_template_dir'] + "/template_align_all.job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['result_dir'] + "/align_all.job", "w") as outfile:
        outfile.write(job_file)


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-dir", required=True)
arg_parser.add_argument("-alpha", required=False, default=0.4, type=float)
arg_parser.add_argument("-p_0", required=False, default=0.2, type=float)
arg_parser.add_argument("-wall_time", required=False, default="05:00:00", type=str)

args = arg_parser.parse_args()

params = parse_config(args)

generate_single_job(**params)
print "Job prepared, but not sent."
