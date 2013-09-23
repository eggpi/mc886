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
pages under a domain based on their average hit rate per page load (the
average number of times a subresource was loaded from each page that requested
it), and the timestamp of the latest load of the subresource.

The former should be a "relative" measure of relevance for a subresource -- how
important it is for the pages that request it -- while the latter is a rough
measure of "absolute" relevance -- how often it gets requested by any page.

Whenever we want to make predictions for a page load, we use the cluster that
contains more subresources that were recently requested by that page.

Unanswered questions:
    - How to pick a suitable K?
    - As an alternative if this doesn't work, maybe we can cluster subresource
      _loads_ instead of subresources?
    - What to do with anomalous subresources that are loaded 58 times for a
      single page load? e.g.: http://i1.ytimg.com/vi/ylWORyToTo4/default.jpg
    - Maybe use subresource hits / all page hits to also get global importance?
'''

ONE_MINUTE = 60 * 1e6
ONE_HOUR = 60 * ONE_MINUTE
ONE_DAY = 24 * ONE_HOUR
ONE_WEEK = 7L * ONE_DAY

EPOCH = datetime(1970, 1, 1)
NOW = datetime(2013, 9, 21, 0, 0, 0, 0) # datetime.now()
SHIFTED_EPOCH =  NOW - relativedelta(weeks = 2)
END_OF_THE_WORLD = datetime.now()

SHIFTED_EPOCH_US = 1e6 * (SHIFTED_EPOCH - EPOCH).total_seconds()
END_OF_THE_WORLD_US = 1e6 * (END_OF_THE_WORLD - EPOCH).total_seconds()

def normalize_timestamp(timestamp):
    return (timestamp - SHIFTED_EPOCH_US) / \
           (END_OF_THE_WORLD_US - SHIFTED_EPOCH_US)

def make_vector_for_subresource(accesses):
    total_hits = 0.0
    page_hits = 0.0
    most_recent_hit = float('-inf')
    for _, page_loads, last_hit, hits in accesses:
        total_hits += hits
        page_hits += page_loads

        if most_recent_hit < last_hit:
            most_recent_hit = last_hit

    hits_per_page_hit = min(1.0, total_hits / (1.0 + page_hits))
    return (hits_per_page_hit, normalize_timestamp(most_recent_hit))

def get_host_for_uri(uri):
    parts = urlparse.urlparse(uri)
    return parts.scheme + '://' + parts.netloc

def cluster_subresources_for_hosts(k, db):
    '''
    Cluster the subresources loaded by all pages under
    a single host using their hit rate and timestamp.
    '''

    # A dictionary mapping hosts to subresource loads by pages under that
    # host. Each subresource load is in turn represented by a dictionary mapping
    # the subresource's url to a list of tuples (page id, last_hit, hits)
    subresources_for_host = defaultdict()
    subresources_for_host.default_factory = lambda: defaultdict(list)

    cursor = db.cursor()
    cursor.execute('select id, uri, loads from moz_pages')
    for pid, uri, page_loads in cursor.fetchall():
        # find host for each page
        host = get_host_for_uri(uri)

        # find the subresources loaded by the page
        cursor.execute(
            'select uri, last_hit, hits from moz_subresources where pid = ?',
            (pid,))

        for uri, last_hit, hits in cursor:
            subresources_for_host[host][uri].append((pid, page_loads, last_hit, hits))

    clusters_for_hosts = {}
    distortions = {}
    for host, host_sres in subresources_for_host.items():
        if len(host_sres) > k:
            sres_uris = host_sres.keys()
            sres_accesses = [host_sres[uri] for uri in sres_uris]
            sres_vectors = np.array(
                [make_vector_for_subresource(acs) for acs in sres_accesses],
                dtype = 'float32')

            distortion, clusters, means = cv2.kmeans(
                sres_vectors,
                K = k,
                criteria = (cv2.TERM_CRITERIA_MAX_ITER, 100, 0), # 100 iterations
                attempts = 20,
                flags = cv2.KMEANS_RANDOM_CENTERS)

            print 'Distortion for host {} is {}'.format(host, distortion)
            distortions[host] = distortion

            # transform clusters into list of tuples:
            # [ (mean, [list of subresources for cluster 0]),
            #   (mean, [list of subresources for cluster 1]), ...
            # ]
            clusters_for_hosts[host] = []
            for mean in means:
                clusters_for_hosts[host].append((mean, []))

            assert len(clusters) == len(sres_uris)
            for c, uri, accesses in zip(clusters, sres_uris, sres_accesses):
                assert c.shape == (1,)
                c = c[0]
                clusters_for_hosts[host][c][1].append((uri, accesses))
        else:
            # FIXME figure this out
            pass

    return clusters_for_hosts, distortions

def predict_for_page_load(db, page, clusters_for_hosts):
    cursor = db.cursor()
    cursor.execute(
        'select uri from moz_subresources where pid = ? and last_hit > ?',
        (page[0], page[3]))

    sres_from_last_time = set(sr[0] for sr in cursor.fetchall())
    host = get_host_for_uri(page[1])
    clusters = clusters_for_hosts[host]

    best_correspondence = 0
    for i, (mean, subresources) in enumerate(clusters):
        in_cluster = set((uri for uri, accesses in subresources))
        correspondence = len(sres_from_last_time & in_cluster)

        if correspondence > best_correspondence:
            best_correspondence = correspondence
            closest = i

    print 'Cluster sizes: {}'.format([len(c[1]) for c in clusters])
    print 'Correspondence: {}'.format(best_correspondence)

    sres = clusters[closest][1]
    return closest, clusters, [sr[0] for sr in sres], sres_from_last_time

def visualize(clusters, closest_cluster, explicit_sres):
    colors = [
        'black',
        'yellow',
        'green',
        'blue',
        'pink',
        'grey'
    ]

    assert len(colors) >= len(clusters) - 1

    for cidx, (mean, sres) in enumerate(clusters):
        color = 'red' if cidx == closest_cluster else colors.pop(0)

        plot.subplot(211)
        print '{0} cluster: {1}'.format(color, mean)
        plot.plot(*mean, color = color, marker = 'v')

        for uri, accesses in sres:
            if uri in explicit_sres:
                plot.subplot(212)
                plot.plot(*make_vector_for_subresource(accesses),
                          color = color, marker = 'o')

            plot.subplot(211)
            plot.plot(*make_vector_for_subresource(accesses),
                      color = color, marker = 'o')

    plot.subplot(211)
    plot.title('Clusters for host')
    plot.xlabel('hits per page load')
    plot.ylabel('normalized timestamp')

    plot.subplot(212)
    plot.title('Subresources from last load')
    plot.xlabel('hits per page load')
    plot.ylabel('normalized timestamp')

    plot.show()

if __name__ == "__main__":
    import sys

    dbfile = sys.argv[1]
    page_uri = sys.argv[2]

    k = 6
    with sqlite3.connect(dbfile) as db:
        clusters_for_hosts, distortions = cluster_subresources_for_hosts(k, db)

        cursor = db.cursor()
        cursor.execute('select * from moz_pages where uri = ?', (page_uri,))
        page = cursor.fetchone()

        cidx, c, predicted_sres, explicit_sres = \
            predict_for_page_load(db, page, clusters_for_hosts)

    print 'Would take predictive actions for {0} items, ' \
          'out of which {1} were explicitly loaded last time, ' \
          'and the page loaded a total of {2} subresources last time' \
          .format(len(predicted_sres),
                  len(set(predicted_sres) & set(explicit_sres)),
                  len(explicit_sres))

    visualize(c, cidx, explicit_sres)
