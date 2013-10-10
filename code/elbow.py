import sqlite3
import matplotlib.pyplot as plot
from moz_kmeans import load_database, cluster_resources_for_host

if __name__ == '__main__':
    import sys

    dbfile = sys.argv[1]
    huri = sys.argv[2]

    krange = range(2, 50)
    distortions = []
    with sqlite3.connect(dbfile) as db:
        hindex, _, rindex = load_database(db)
        host = hindex[huri]

    for k in krange:
        distortions.append(cluster_resources_for_host(host, rindex, k, 0))

    plot.plot(krange, distortions, 'b-')
    plot.show()
