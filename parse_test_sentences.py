import re
import sys


outfile = open(sys.argv[1] +".sent", "w")
with open(sys.argv[1], "r") as infile:
    for line in infile:
        m = re.search("<seg .*?>(.*?)</seg>", line)
        if m:
            outfile.write(m.group(1) + "\n")

outfile.close()

