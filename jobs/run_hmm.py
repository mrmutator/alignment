import argparse
import time
import os

def get_time():
    return str(int(time.time()))

def make_directories(dir, num_iterations):
    for i in xrange(1, num_iterations + 1):
        path_it = dir + "/it" + str(i) + "/"
        path_job = dir  + "/jobs" + str(i) + "/"
        if not os.path.exists(path_it):
            os.makedirs(path_it)
        if not os.path.exists(path_job):
            os.makedirs(path_job)


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-dir", required=True)
arg_parser.add_argument("-job_name", required=False, default=get_time())

arg_parser.add_argument("-e", required=True)
arg_parser.add_argument("-f", required=True)

arg_parser.add_argument("-num_iterations", required=True, type=int)

arg_parser.add_argument("-alpha", required=False, default=0.0, type=float)
arg_parser.add_argument("-p0", required=False, default=0.2, type=float)

arg_parser.add_argument("-num_nodes", required=False, default=0, type=int)
arg_parser.add_argument("-num_workers", required=False, default=16, type=int)

args = arg_parser.parse_args()

dir = args.dir.rstrip("/")
num_iterations = args.num_iterations
e_path = args.e
f_path = args.f

# make directories
make_directories(dir, num_iterations)




# To-Do
# smart way to automatically define number of nodes
