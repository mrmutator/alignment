import codecs
import sys
import re
from HTMLParser import HTMLParser


h = HTMLParser()
c = 0
outfile = codecs.open(sys.argv[1] + ".normalized", "w", "utf-8")
with open(sys.argv[1], "r") as infile:
    for line in infile:
        c += 1
        line = line.decode("utf-8")
        line = line.strip()
        line = re.sub("_", "##undl##", line)
        line = re.sub("googletag.*?\}\);", "", line)
        line = h.unescape(line)
        line = " ".join(line.split())
        outfile.write(line + "\n")
outfile.close()
print c

