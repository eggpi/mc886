import random
import pprint

M = 5 # dimension, number of weights
N = 100 # number of measurements

# make a linear function given weights
def make_f(weights):
    return lambda x: sum(t * s for t, s in zip(weights, x))

# make a random set of data points to be fed to y,
# that is, a vector of M numbers, the first being
# 1.0 by definition.
def make_x():
    return [1.0] + random.sample(range(1000), M - 1)

def J(f):
    return sum((f(x) - y)**2 for x, y in data) / (2 * N)

# the partial derivative of function J(w) for the i-th weight,
# f is the current estimation function (that is, make_f(w))
def DJ(f, i):
    return sum((f(x) - y) * x[i] for x, y in data) / N

# calculate the norm of vector v
def norm(v):
    return sum(x**2 for x in v)**0.5

# create the target function and data points
target = []
for i in range(M):
    target.append(random.random())
y = make_f(target)

x = []
for i in range(N):
    x.append(make_x())

# min-max normalization
xnorms = [norm(v) for v in x]
mi, ma = min(xnorms), max(xnorms)

data = []
for v in x:
    v = [(e - mi) / (ma - mi) for e in v]
    data.append((v, y(v)))

print "Data: "
pprint.pprint(data)

# initial guess at parameters
learned = []
for i in range(M):
    learned.append(random.random())

alpha = 0.0001
err = float('inf')
while err > 0.0001:
    print "Starting iteration!"
    print "    :: learned = {0}".format(learned)
    print "    :: J(learned) = {0}".format(J(make_f(learned)))

    f = make_f(tuple(learned))
    for i in range(M):
        learned[i] = learned[i] - alpha * DJ(f, i)

    err = J(make_f(learned))

    print "    :: learned = {0}".format(learned)
    print "    :: err = {0}".format(err)
    print "End iteration!"

print "Learned is: {0}".format(learned)
print "Target was: {0}".format(target)
