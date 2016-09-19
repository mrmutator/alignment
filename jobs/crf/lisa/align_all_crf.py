import argparse
import os
import glob



def parse_config(args):

    params = dict()
    params['dir'] = os.path.abspath(args.dir)
    params['fvoc'] = os.path.abspath(glob.glob(os.path.join(params["dir"], "*.fvoc"))[0])
    params['fvoc'] = os.path.abspath(glob.glob(os.path.join(params["dir"], "*.weights"))[0])
    params['corpus'] = os.path.abspath(args.corpus)
    params['corpus_dir'] = os.path.dirname(os.path.realpath(params['corpus']))
    params['ibm1_table'] = os.path.abspath(os.path.join(params['corpus_dir'], 'ibm1.table'))
    params['e_voc'] = os.path.abspath(os.path.join(params['corpus_dir'], 'e.vcb'))
    params['f_voc'] = os.path.abspath(os.path.join(params['corpus_dir'], 'f.vcb'))
    params['num_nodes'] = int(args.num_nodes)
    params['job_template_dir'] = os.path.dirname(os.path.realpath(__file__))
    params['script_dir'] = os.path.abspath(os.path.join(params['job_template_dir'], '../../../'))
    params['num_workers'] = 16

    params['wall_time'] = args.wall_time

    return params


def generate_single_job(**params):
    with open(params['job_template_dir'] + "/template_align_all.job.txt", "r") as infile:
        template = infile.read()
        job_file = template % params
    with open(params['dir'] + "/align_all.job", "w") as outfile:
        outfile.write(job_file)


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-dir", required=True)
arg_parser.add_argument("-corpus", required=True)
arg_parser.add_argument("-num_nodes", required=False, type=int, default=1)
arg_parser.add_argument("-wall_time", required=False, default="05:00:00", type=str)

args = arg_parser.parse_args()

params = parse_config(args)

generate_single_job(**params)
print "Job prepared, but not sent."
