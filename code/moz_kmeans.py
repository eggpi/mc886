import time
import pprint
import sqlite3
from urllib2 import urlparse
from collections import defaultdict, Counter
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

K = 8
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

bias_for_host = {}
def make_vector_for_subresource(host, accesses):
    total_hits = 0.0
    page_hits = 0.0
    most_recent_hit = float('-inf')
    for _, page_loads, last_hit, hits in accesses:
        total_hits += hits
        page_hits += page_loads

        if most_recent_hit < last_hit:
            most_recent_hit = last_hit

    hits_per_page_hit = min(1.0, total_hits / (bias_for_host[host] + page_hits))
    return (hits_per_page_hit, normalize_timestamp(most_recent_hit))

def get_host_for_uri(uri):
    parts = urlparse.urlparse(uri)
    return parts.scheme + '://' + parts.netloc

def cluster_subresources_for_hosts(k, db):
    '''
    Cluster the subresources loaded by all pages under
    a single host using their hit rate and timestamp.
    '''

    subk = k

    # A dictionary mapping hosts to subresource loads by pages under that
    # host. Each subresource load is in turn represented by a dictionary mapping
    # the subresource's url to a list of tuples (page id, last_hit, hits)
    subresources_for_host = defaultdict()
    subresources_for_host.default_factory = lambda: defaultdict(list)

    page_hits_for_host = defaultdict(Counter)

    cursor = db.cursor()
    cursor.execute('select id, uri, loads from moz_pages')
    for pid, uri, page_loads in cursor.fetchall():
        # find host for each page
        host = get_host_for_uri(uri)
        page_hits_for_host[host][page_loads] += 1

        # find the subresources loaded by the page
        cursor.execute(
            'select uri, last_hit, hits from moz_subresources where pid = ?',
            (pid,))

        for uri, last_hit, hits in cursor:
            subresources_for_host[host][uri].append((pid, page_loads, last_hit, hits))

    global bias_for_host
    for host, counter in page_hits_for_host.items():
        bias, count = counter.most_common()[0]
        bias_for_host[host] = bias
        print 'Bias for {} is {}'.format(host, bias_for_host[host])

    clusters_for_hosts = {}
    subclusters_for_clusters = {} # host -> cluster -> list of uris
    distortions = {}
    for host, host_sres in subresources_for_host.items():
        if len(host_sres) > k:
            sres_uris = host_sres.keys()
            sres_accesses = [host_sres[uri] for uri in sres_uris]
            sres_vectors = np.array(
                [make_vector_for_subresource(host, acs)
                 for uri, acs in zip(sres_uris, sres_accesses)],
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

            # compute subclusters for each cluster
            subclusters_for_clusters[host] = {}
            for c, (mean, sres) in enumerate(clusters_for_hosts[host]):
                if len(sres) <= subk:
                    continue

                cluster_sres_vectors = np.array(
                    [make_vector_for_subresource(host, acs) for uri, acs in sres],
                    dtype = 'float32')

                _, subcl, means = cv2.kmeans(
                    cluster_sres_vectors,
                    K = subk,
                    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 100, 0), # 10 iterations
                    attempts = 20,
                    flags = cv2.KMEANS_RANDOM_CENTERS)

                subclusters_for_clusters[host][c] = []
                for mean in means:
                    subclusters_for_clusters[host][c].append((mean, []))

                for sc, (uri, accesses) in zip(subcl, sres):
                    assert sc.shape == (1,)
                    sc = sc[0]
                    subclusters_for_clusters[host][c][sc][1].append(uri)

                cluster_size = len(clusters_for_hosts[host][c][1])
                subcluster_sizes = map(len, (subclusters_for_clusters[host][c][sc][1] for sc in range(subk)))
                assert cluster_size == sum(subcluster_sizes)

                print cluster_size, subcluster_sizes
        else:
            # FIXME figure this out
            pass

    return clusters_for_hosts, subclusters_for_clusters, distortions

def predict_for_page_load(db, page, clusters_for_hosts, subclusters_for_clusters):
    cursor = db.cursor()
    cursor.execute(
        'select uri from moz_subresources where pid = ? and last_hit > ?',
        (page[0], page[3]))

    sres_from_last_time = set(sr[0] for sr in cursor.fetchall())
    host = get_host_for_uri(page[1])
    clusters = clusters_for_hosts[host]
    subclusters = subclusters_for_clusters[host]

    best_correspondence = set()
    for i, (mean, subresources) in enumerate(clusters):
        in_cluster = set((uri for uri, accesses in subresources))
        correspondence = sres_from_last_time & in_cluster

        if len(correspondence) > len(best_correspondence):
            best_correspondence = correspondence
            closest = i

    if len(best_correspondence) == 0:
        return None

    print 'Cluster size: {}'.format(len(clusters[closest][1]))
    print 'Correspondence: {}'.format(len(best_correspondence))

    subclusters = subclusters[closest]

    predicted_sres = set()
    corresponded_and_picked = 0
    min_corresponded_picked = 0.5 * len(best_correspondence)
    max_predicted_size = 2 * len(best_correspondence)
    for mean, uris in subclusters:
        corresponded_in_subcluster = len(set(uris) & best_correspondence)
        if not corresponded_in_subcluster:
            continue

        too_big = len(predicted_sres) + len(uris) > max_predicted_size
        has_min_picked = corresponded_and_picked >= min_corresponded_picked

        if not has_min_picked or not too_big:
            predicted_sres = predicted_sres.union(uris)
            corresponded_and_picked += corresponded_in_subcluster
        else:
            break

    return closest, clusters, predicted_sres, sres_from_last_time

def visualize(clusters, host, closest_cluster, predicted_sres, explicit_sres):
    colors = [
        'black',
        'yellow',
        'green',
        'blue',
        'pink',
        'grey',
        'orange'
    ]

    assert len(colors) >= len(clusters) - 1

    for cidx, (mean, sres) in enumerate(clusters):
        if cidx == closest_cluster:
            plot.subplot(223)
            for uri, accesses in sres:
                if uri in predicted_sres:
                    color = 'red'
                elif uri in explicit_sres:
                    color = 'orange'
                else:
                    color = 'black'

                plot.plot(*make_vector_for_subresource(host, accesses),
                          color = color, marker = 'o')

        color = 'red' if cidx == closest_cluster else colors.pop(0)

        plot.subplot(221)
        print '{0} cluster: {1}'.format(color, mean)
        plot.plot(*mean, color = color, marker = 'v')

        examples = 3
        for uri, accesses in sres:
            if examples > 0 and len(uri) < 80:
                examples -= 1
                print '    {}'.format(uri)

            if uri in explicit_sres:
                plot.subplot(222)
                plot.plot(*make_vector_for_subresource(host, accesses),
                          color = color, marker = 'o')

            plot.subplot(221)
            plot.plot(*make_vector_for_subresource(host, accesses),
                      color = color, marker = 'o')

    plot.subplot(221)
    plot.title('Clusters for host (red is the selected cluster)')
    plot.xlabel('hits per page load')
    plot.ylabel('normalized timestamp')

    plot.subplot(222)
    plot.title('Subresources from last load (colored by cluster)')
    plot.xlabel('hits per page load')
    plot.ylabel('normalized timestamp')

    plot.subplot(223)
    plot.title('Selected cluster (red are predicted subresources)')
    plot.xlabel('hits per page load')
    plot.ylabel('normalized timestamp')


    plot.show()

if __name__ == "__main__":
    import sys

    dbfile = sys.argv[1]
    page_uri = sys.argv[2]

    with sqlite3.connect(dbfile) as db:
        clusters_for_hosts, subclusters_for_clusters, distortions = cluster_subresources_for_hosts(K, db)

        cursor = db.cursor()
        cursor.execute('select * from moz_pages where uri = ?', (page_uri,))
        page = cursor.fetchone()

        cidx, c, predicted_sres, explicit_sres = \
            predict_for_page_load(db, page, clusters_for_hosts, subclusters_for_clusters)

    print 'Would take predictive actions for {0} items, ' \
          'out of which {1} were explicitly loaded last time, ' \
          'and the page loaded a total of {2} subresources last time' \
          .format(len(predicted_sres),
                  len(set(predicted_sres) & set(explicit_sres)),
                  len(explicit_sres))

    visualize(c, get_host_for_uri(page_uri), cidx, predicted_sres, explicit_sres)
