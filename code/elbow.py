import sqlite3
import matplotlib.pyplot as plot
from moz_kmeans import cluster_subresources_for_hosts

if __name__ == '__main__':
    import sys

    dbfile = sys.argv[1]
    host = sys.argv[2]

    krange = range(1, 20)
    distortions = []
    with sqlite3.connect(dbfile) as db:
        for k in krange:
            _, _, d = cluster_subresources_for_hosts(k, db)
            distortions.append(d[host])

        plot.plot(krange, distortions, 'b-')

    plot.show()
