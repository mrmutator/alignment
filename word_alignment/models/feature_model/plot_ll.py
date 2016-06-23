from matplotlib import pyplot as plt
import sys

def read_data(fname):
    ll = []
    with open(fname) as infile:
        for line in infile:
           ll.append(float(line.split()[0]))
    return ll


d1 = read_data(sys.argv[1])[50:]
d2 = read_data(sys.argv[2])[50:]


plt.plot(d1, label="d1")
plt.plot(d2, label="d2")
plt.legend()
plt.show()