import subprocess
import os

def read_job_file(file_name):
    jobs = dict()
    with open(file_name, "r") as infile:
        for line in infile:
            job_id, job_file, array, dep = line.split()
            job_id = int(job_id)
            if array == "-":
                array = None
            if dep == "-":
                dep = None
            assert job_id not in jobs
            jobs[job_id] = (job_file, array, dep)
    return jobs

def send_jobs(jobs, log_file_name):
    with open(log_file_name, "w") as outfile:
        system_job_ids = dict()
        for job_id in sorted(jobs):
            job_file, array, dep = jobs[job_id]
            if job_id < 0:
                system_job_ids[job_id] = job_file
                continue

            depend_string = []
            if dep:
                if jobs[dep][1]:
                    depend_string = ["-Wdepend=afterokarray:" + system_job_ids[dep]]
                else:
                    depend_string = ["-Wdepend=afterok:" + system_job_ids[dep]]
            array_string = []
            if array:
                array_string = ["-t", array]

            job_dir = os.path.dirname(os.path.realpath(job_file))
            command = ['qsub'] + depend_string + array_string + [job_file]
            print "Executing ", " ".join(command)
            proc_job = subprocess.Popen(command, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, cwd=job_dir)
            stdout, stderr = proc_job.communicate()
            if stderr:
                raise Exception("Failed sending job " + str(job_id) + " : " + stderr)
            system_job_id = stdout.strip().split(".")[0]
            system_job_ids[job_id] = system_job_id
            outfile.write(" ".join([system_job_id, job_id] + command) + "\n")
    print "All jobs sent."
    print "Check logfile ", log_file_name


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("joblist")
    args = arg_parser.parse_args()

    jobs = read_job_file(args.joblist)
    send_jobs(jobs, args.joblist + ".log")