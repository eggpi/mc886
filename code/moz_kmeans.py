import time
import pprint
import sqlite3
from urllib2 import urlparse
from collections import defaultdict
from datetime import datetime
from dateutil.relativedelta import relativedelta

import cv2
import numpy as np
import matplotlib.pyplot as plot

'''
A sketch for prediction based on k-means.

The basic idea is to use k-means to cluster subresources fetched by the
pages under a domain based on their number of hits and timestamp of the last
hit. These clusters should roughly correspond to the pages, or groups of pages
accessed close together.

Whenever we want to predict a new page load, we look for the cluster to which
that page fits more closely (in the future the page should actually belong to a
cluster), and take predictive actions for everything there.

Unanswered questions:
    - How to pick a suitable K?
    - We need to normalize the data for k-means to work well. How can we
      normalize a timestamp?
    - Are we going to run k-means on every prediction request (that would make
      normalization easy)? How expensive is that? Look into online kmeans!
'''

ONE_MINUTE = 60 * 1e6
ONE_HOUR = 60 * ONE_MINUTE
ONE_DAY = 24 * ONE_HOUR
ONE_WEEK = 7L * ONE_DAY

K = 5 # FIXME needs tweaking

EPOCH = datetime(1970, 1, 1)
SHIFTED_EPOCH = datetime(2013, 1, 1)
END_OF_THE_WORLD = SHIFTED_EPOCH + relativedelta(years = 1)

SHIFTED_EPOCH_US = 1e6 * (SHIFTED_EPOCH - EPOCH).total_seconds()
END_OF_THE_WORLD_US = 1e6 * (END_OF_THE_WORLD - EPOCH).total_seconds()

def make_vector(hits, timestamp):
    ts = (timestamp - SHIFTED_EPOCH_US) / \
         (END_OF_THE_WORLD_US - SHIFTED_EPOCH_US)
    return hits, ts

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
                [make_vector(sres[-2], sres[-1]) for sres in host_sres],
                dtype = 'float32')

            retval, clusters, means = cv2.kmeans(
                sres_vectors,
                K = K,
                criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10, 0), # 10 iterations
                attempts = 3,
                flags = cv2.KMEANS_RANDOM_CENTERS)

            # transform clusters into list of tuples:
            # [ (mean, [list of subresources for cluster 0]),
            #   (mean, [list of subresources for cluster 1]), ...
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

    print 'Cluster sizes: {}'.format([len(c[1]) for c in clusters])

    means = [c[0] for c in clusters]
    distances = map(
        lambda mean: np.linalg.norm(mean - make_vector(page[-2], page[-1])),
        means)

    # find the closest cluster
    closest = None
    mindist = float('inf')
    for i, d in enumerate(distances):
        if d < mindist:
            closest, mindist = i, d

    sres = clusters[closest][1]
    return closest, clusters, sres

def visualize(page, clusters, closest_cluster, explicit_sres):
    colors = [
        'black',
        'yellow',
        'green',
        'blue'
    ]

    assert len(colors) >= K - 1

    page_coords = make_vector(page[-2], page[-1])

    print 'Page is at: {0}'.format(page_coords)
    plot.plot(*page_coords, color = 'brown', marker = '^')
    plot.annotate('loaded page', xy = page_coords,
                  textcoords = 'axes fraction',
                  xytext = (0.2, 0.8),
                  arrowprops = {
                      'facecolor': 'black',
                      'shrink': 0.04,
                      'width': 1,
                      'headwidth': 10
                  })

    for cidx, (mean, sres) in enumerate(clusters):
        color = 'red' if cidx == closest_cluster else colors.pop(0)

        print '{0} cluster: {1}'.format(color, mean)
        plot.plot(*mean, color = color, marker = 'v')
        for sr in sres:
            plot.plot(*make_vector(sr[-2], sr[-1]), color = color, marker = 'o')

    plot.title('Clusters for host')
    plot.xlabel('hits')
    plot.ylabel('normalized timestamp')

    for sr in explicit_sres:
        plot.plot(*make_vector(sr[-2], sr[-1]), color = 'pink', marker = 'o')

    plot.show()

if __name__ == "__main__":
    import sys

    hosts, pages, subresources = load_db(sys.argv[1])
    clusters_for_hosts = cluster_subresources_for_hosts(hosts, pages, subresources)

    page = find_page_by_uri(sys.argv[2], pages)
    explicit_sres = [sr for sr in subresources if sr[1] == page[0]]

    cidx, c, predicted_sres = \
        predict_for_page_load(sys.argv[2], pages, clusters_for_hosts, hosts)

    print 'Would take predictive actions for {0} items, ' \
          'out of which {1} were explicitly loaded last time, ' \
          'and the page loaded a total of {2} subresources' \
          .format(len(predicted_sres),
                  len(set(predicted_sres) & set(explicit_sres)),
                  len(explicit_sres))

    visualize(page, c, cidx, explicit_sres)
