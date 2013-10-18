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

K = 10
C = 3
EPOCH = datetime(1970, 1, 1)
NOW = datetime(2013, 9, 21, 0, 0, 0, 0) # datetime.now()

WINDOW_START =  NOW - relativedelta(weeks = 2)
WINDOW_START_US = 1e6 * (WINDOW_START - EPOCH).total_seconds()

WINDOW_END = NOW
WINDOW_END_US = 1e6 * (WINDOW_END - EPOCH).total_seconds()

def get_host_for_uri(uri):
    parts = urlparse.urlparse(uri)
    return parts.scheme + '://' + parts.netloc

def normalize_timestamp(timestamp):
    return (timestamp - WINDOW_START_US) / (WINDOW_END_US - WINDOW_START_US)

class Host(object):
    def __init__(self, name):
        self.name = name
        self.pages = []
        self.bias = 0.0
        self.clusters = []
        self.subclusters = []

    def add_page(self, page):
        self.pages.append(page)

    def compute_bias(self):
        page_hits = Counter(page.hits for page in self.pages)
        self.bias, count = page_hits.most_common()[0]

class Page(object):
    def __init__(self, uri, hits, last_hit, resource_loads):
        self.uri = uri
        self.hits = hits
        self.last_hit = last_hit
        self.host = get_host_for_uri(self.uri)
        self.resource_loads = resource_loads # (uri, hits, last_hit)

    def get_resources_from_last_load(self):
        return [uri
                for uri, hits, last_hit in self.resource_loads
                if last_hit >= self.last_hit]

class Resource(object):
    def __init__(self, uri):
        self.uri = uri
        # filter loads per host
        self.loading_pages_per_host = defaultdict(list)

    def add_loading_page(self, page, hits, last_hit):
        # page must not exist yet
        assert all(
            page.uri != p.uri
            for p, _, _ in self.loading_pages_per_host[page.host])

        self.loading_pages_per_host[page.host].append((page, hits, last_hit))

    def get_fv_for_host(self, host):
        # find all loads by pages of this host
        loads = self.loading_pages_per_host[host.name]

        # use these loads to compute feature vector
        total_hits = sum(hits for page, hits, last_hit in loads)
        page_hits = sum(page.hits for page, hits, last_hit in loads)
        last_hit = max(last_hit for page, hits, last_hit in loads)

        # biased resource hits per loading page hit
        hits_per_page_hit = min(1.0, float(total_hits) / (host.bias + page_hits))

        # timestamp of last hit
        normalized_last_hit = normalize_timestamp(last_hit)

        return (hits_per_page_hit, normalized_last_hit)

def load_database(db):
    '''
    Load data from the database. Returns the hosts, pages and resources in the
    database as dictionaries keyed by their uris.
    '''

    hosts = {}
    pages = {}
    resources = {}

    cursor = db.cursor()
    cursor.execute('select id, uri, loads, last_load from moz_pages')
    for pid, puri, page_loads, last_load in cursor.fetchall():
        # find host for each page, and create the
        # corresponding object if necessary
        huri = get_host_for_uri(puri)
        host = hosts.setdefault(huri, Host(huri))

        # find the resources loads by the page
        cursor.execute(
            'select uri, hits, last_hit from moz_subresources where pid = ?',
            (pid,))
        resource_loads_for_page = list(cursor)

        assert puri not in pages
        page = Page(puri, page_loads, last_load, resource_loads_for_page)
        pages[puri] = page

        for ruri, hits, last_hit in resource_loads_for_page:
            resource = resources.setdefault(ruri, Resource(ruri))
            resource.add_loading_page(page, hits, last_hit)

        host.add_page(page)

    # compute the bias for each host
    for host in hosts.values():
        host.compute_bias()

    return hosts, pages, resources

def build_clusters(means, clusters, resources):
    # transform clusters into a list of (mean, [resources])
    cluster = map(lambda m: (tuple(m), []), means)
    for c, r in zip(clusters, resources):
        assert c.shape == (1,)
        c = c[0]
        cluster[c][1].append(r)
    return cluster

def cluster_resources_for_host(host, rindex, k = K, subk = K):
    # find all resources needed by all pages under the host
    host_ruris = set(
        rl[0]
        for page in host.pages
        for rl in page.resource_loads
    )

    host_resources = [rindex[ruri] for ruri in host_ruris]
    fvs = [r.get_fv_for_host(host) for r in host_resources]

    kmeans_criteria = (cv2.TERM_CRITERIA_MAX_ITER, 100, 0) # 100 iterations

    # top-level clustering
    top_distortion, top_clusters, top_means = cv2.kmeans(
        np.array(fvs, dtype = 'float32'),
        K = k,
        criteria = kmeans_criteria,
        attempts = 20,
        flags = cv2.KMEANS_RANDOM_CENTERS)

    clusters = build_clusters(top_means, top_clusters, host_resources)

    # subclusters
    subclusters = []
    if subk > 0:
        for mean, cluster_resources in clusters:
            fvs = [r.get_fv_for_host(host) for r in cluster_resources]

            sub_distortion, sub_clusters, sub_means = cv2.kmeans(
                np.array(fvs, dtype = 'float32'),
                K = subk,
                criteria = kmeans_criteria,
                attempts = 20,
                flags = cv2.KMEANS_RANDOM_CENTERS)

            subclusters.append(
                build_clusters(sub_means, sub_clusters, cluster_resources)
            )

    host.clusters = clusters
    host.subclusters = subclusters

    # print clusters and subclusters for debugging
    for i, (mean, resources) in enumerate(host.clusters):
        print '{} {} {}:'.format(i, mean, len(resources)),
        for mean, resources in host.subclusters[i]:
            print len(resources),
        print

    return top_distortion

def pick_best_cover(resources_to_cover, clusters, cover = C):
    '''
    Given a list of resources to cover and some clusters,
    return a list of at most `cover` tuples representing the
    clusters that best cover the list of resources.
    In each tuple, the first element is an index in `clusters`, and
    the second is that cluster's correspondence.
    '''

    best_clusters = [(-1, set())] * cover
    for i, (mean, resources) in enumerate(clusters):
        in_cluster = [resource.uri for resource in resources]
        correspondence = set(resources_to_cover) & set(in_cluster)

        if len(correspondence) > len(best_clusters[0][1]):
            best_clusters[0] = (i, correspondence)
            best_clusters.sort(key = lambda e: len(e[1]))

    return filter(lambda (c, corr): c > -1, best_clusters)

def predict_for_page_load(page, hindex):
    host = hindex[page.host]

    # cover resources from last load with clusters
    clusters, subclusters = host.clusters, host.subclusters
    res_from_last_load = page.get_resources_from_last_load()
    if not res_from_last_load:
        return None

    cover_clusters = pick_best_cover(res_from_last_load, clusters)

    print 'Cover clusters: {}'.format([i for i, _ in cover_clusters])

    # compute the list of all subclusters in all of the best clusters,
    # and all the resources that have been covered
    to_cover_subclusters = []
    covered_resources = set()
    for cluster, correspondence in cover_clusters:
        to_cover_subclusters += subclusters[cluster]
        covered_resources = covered_resources.union(correspondence)

    cover_subclusters = pick_best_cover(covered_resources, to_cover_subclusters)
    print 'Cover subclusters: {}, sizes {}'.format(
        [i for i, _ in cover_subclusters],
        [len(to_cover_subclusters[i][1]) for i, _ in cover_subclusters]
    )

    # pick some subclusters to grow our set of predicted resources,
    # but never more than twice the size of the covered resources
    predicted = covered_resources
    for i, correspondence in cover_subclusters:
        ruris = [resource.uri for resource in to_cover_subclusters[i][1]]
        with_subcluster = predicted.union(ruris)

        if len(with_subcluster) < 2 * len(covered_resources):
            predicted = predicted.union(ruris)

    return predicted

def visualize(host, closest_cluster, predicted, explicit):
    colors = [
        'black',
        'yellow',
        'green',
        'blue',
        'pink',
        'grey',
        'orange',
        'cyan',
        'magenta'
    ]

    clusters = host.clusters
    assert len(colors) >= len(clusters) - 1

    for cidx, (mean, resources) in enumerate(clusters):
        if cidx == closest_cluster:
            plot.subplot(223)
            for res in resources:
                if res.uri in predicted:
                    color = 'red'
                elif res.uri in explicit:
                    color = 'orange'
                else:
                    color = 'black'

                plot.plot(*res.get_fv_for_host(host),
                          color = color, marker = 'o')

        color = 'red' if cidx == closest_cluster else colors.pop(0)

        plot.subplot(221)
        print '{0} cluster: {1}'.format(color, mean)
        plot.plot(*mean, color = color, marker = 'v')

        examples = 3
        for res in resources:
            if examples > 0 and len(res.uri) < 80:
                examples -= 1
                print '    {}'.format(res.uri)

            if res.uri in explicit:
                plot.subplot(222)
                plot.plot(*res.get_fv_for_host(host),
                          color = color, marker = 'o')

            plot.subplot(221)
            plot.plot(*res.get_fv_for_host(host),
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
        hindex, pindex, rindex = load_database(db)

    huri = get_host_for_uri(page_uri)
    host = hindex[huri]
    cluster_resources_for_host(host, rindex)

    page = pindex.get(page_uri)
    if page is None:
        print >>sys.stderr, 'Unknown page'
        sys.exit(1)

    explicit = page.get_resources_from_last_load()
    predicted = predict_for_page_load(page, hindex)

    explicit_predicted = set(predicted) & set(explicit)
    print 'Would take predictive actions for {0} resources, ' \
          'out of which {1} were explicitly loaded last time, ' \
          'and the page loaded a total of {2} resources last time ' \
          '(so {3:.2f}% of explicit resources were predicted)' \
          .format(len(predicted),
                  len(explicit_predicted),
                  len(explicit),
                  (100.0 * len(explicit_predicted)) / len(explicit))

    #visualize(host, closest, predicted, explicit)
