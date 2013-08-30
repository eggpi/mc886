import pprint
import sqlite3
from urllib2 import urlparse
from collections import defaultdict

import cv2
import numpy as np

K = 3 # FIXME needs tweaking

def load_db(dbfile):
    db = sqlite3.connect(dbfile)
    cursor = db.cursor()

    cursor.execute('SELECT * FROM moz_hosts;')
    hosts = cursor.fetchall()

    cursor.execute('SELECT * FROM moz_subresources;')
    subresources = cursor.fetchall()

    cursor.execute('SELECT * FROM moz_pages;')
    pages = cursor.fetchall()

    db.close()
    return hosts, pages, subresources

# XXX these should be SQL...
def find_host_by_origin(origin, hosts):
    for h in hosts:
        if h[1] == origin:
            return h[0]

    assert False

def find_page_by_uri(uri, pages):
    for p in pages:
        if p[1] == uri:
            return p

    assert False

def get_host_for_uri(uri):
    parts = urlparse.urlparse(uri)
    return parts.scheme + '://' + parts.netloc

def cluster_subresources_for_hosts(hosts, pages, subresources):
    '''
    Cluster the subresources loaded by all pages under
    a single host using their hit rate and timestamp.
    '''

    host_for_page = {}
    for page in pages:
        host = get_host_for_uri(page[1])
        host_for_page[page[0]] = find_host_by_origin(host, hosts)

    subresources_for_host = defaultdict()
    subresources_for_host.default_factory = list
    for sres in subresources:
        page = sres[1]
        host = host_for_page[page]

        subresources_for_host[host].append(sres)

    clusters_for_hosts = {}
    for host, host_sres in subresources_for_host.items():
        if len(host_sres) > K:
            # make a numpy array out of the hit count and
            # timestamp for each subresource
            sres_vectors = np.array(
                [sres[-2:] for sres in host_sres],
                dtype = 'float32')

            _, clusters, means = cv2.kmeans(
                sres_vectors,
                K = K,
                criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10, 0), # 10 iterations
                attempts = 3,
                flags = cv2.KMEANS_RANDOM_CENTERS)

            # transform clusters into list of tuples:
            # [ (mean, list of subresources for cluster 0),
            #   (mean, list of subresources for cluster 1), ...
            # ]
            clusters_for_hosts[host] = []
            for mean in means:
                clusters_for_hosts[host].append((mean, []))

            for c, sres in zip(clusters, host_sres):
                assert c.shape == (1,)
                c = c[0]
                clusters_for_hosts[host][c][1].append(sres)
        else:
            # FIXME figure this out
            pass

    return clusters_for_hosts

def predict_for_page_load(uri, pages, clusters_for_hosts, hosts):
    '''
    Given a page that's being loaded, look up the cluster where it fits
    best among the clusters of subresources under the same host, and take
    predictive actions for the subresources in that cluster.
    '''

    host = find_host_by_origin(get_host_for_uri(uri), hosts)
    clusters = clusters_for_hosts[host]
    page = find_page_by_uri(uri, pages)

    means = [c[0] for c in clusters]
    distances = map(lambda mean: np.linalg.norm(mean - page[-2:]), means)

    # find the closest cluster
    closest = None
    mindist = float('inf')
    for i, d in enumerate(distances):
        if d < mindist:
            closest, mindist = i, d

    sres = clusters[closest][1]

    print 'Would take predictive actions for:'
    pprint.pprint([sr[2] for sr in sres])

if __name__ == "__main__":
    import sys

    hosts, pages, subresources = load_db(sys.argv[1])
    clusters_for_hosts = cluster_subresources_for_hosts(hosts, pages, subresources)

    predict_for_page_load(
        "http://docs.scipy.org/doc/scipy/reference/cluster.vq.html",
        pages, clusters_for_hosts, hosts)
