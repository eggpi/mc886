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

CLUSTER_COVER_SIZE = 3
SUBCLUSTER_COVER_SIZE = 15
K = 10
SUBK = 5
EPOCH = datetime(1970, 1, 1)
NOW = datetime.utcnow()
SLEEP_TIME_SECONDS = 60

def get_host_for_uri(uri):
    parts = urlparse.urlparse(uri)
    return parts.scheme + '://' + parts.netloc

def normalize_timestamp(timestamp):
    window_start =  NOW - relativedelta(weeks = 2)
    window_start_us = 1e6 * (window_start - EPOCH).total_seconds()

    window_end = NOW
    window_end_us = 1e6 * (window_end - EPOCH).total_seconds()

    assert window_end_us > timestamp, "%d > %d" % (timestamp, window_end_us)
    return (timestamp - window_start_us) / (window_end_us - window_start_us)

class Host(object):
    def __init__(self, id, name):
        self.id = id
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
    def __init__(self, id, uri, hits, last_hit, resource_loads):
        self.id = id
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
    cursor.execute('select id, origin from moz_hosts')
    for hid, origin in cursor.fetchall():
        hosts[origin] = Host(hid, origin)

    cursor.execute('select id, uri, loads, last_load from moz_pages')
    for pid, puri, page_loads, last_load in cursor.fetchall():
        # find host for each page
        huri = get_host_for_uri(puri)
        host = hosts[huri]

        # find the resources loads by the page
        cursor.execute(
            'select uri, hits, last_hit from moz_subresources where pid = ?',
            (pid,))
        resource_loads_for_page = list(cursor)

        assert puri not in pages
        page = Page(pid, puri, page_loads, last_load, resource_loads_for_page)
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

def cluster_resources_for_host(host, rindex):
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
    if len(fvs) < K:
        return None

    top_distortion, top_clusters, top_means = cv2.kmeans(
        np.array(fvs, dtype = 'float32'),
        K = K,
        criteria = kmeans_criteria,
        attempts = 20,
        flags = cv2.KMEANS_RANDOM_CENTERS)

    clusters = build_clusters(top_means, top_clusters, host_resources)

    # subclusters
    subclusters = []
    for mean, cluster_resources in clusters:
        fvs = [r.get_fv_for_host(host) for r in cluster_resources]

        sub_distortion, sub_clusters, sub_means = cv2.kmeans(
            np.array(fvs, dtype = 'float32'),
            K = SUBK if SUBK < len(fvs) else 1,
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

def find_most_important_clusters(clusters, n):
    '''
    Given a list of clusters, pick the n most important, as defined
    to be the ones with means closest to (1, 1).
    Returns a list of n tuples containing the cluster indices and importances.
    '''

    if not clusters:
        return None

    most_important = [(-1, float('-inf'))] * n
    for i, (mean, resources) in enumerate(clusters):
        importance = -np.linalg.norm(np.array(mean) - np.array((1, 1)))
        if importance >= most_important[0][1]:
            most_important[0] = (i, importance)
            most_important.sort(key = lambda e: (e[1], e[0] != -1))

    return most_important

def pick_best_cover(resources_to_cover, clusters, cover):
    '''
    Given a list of resources to cover and some clusters,
    return a list of at most `cover` tuples representing the
    clusters that best cover the list of resources.
    In each tuple, the first element is an index in `clusters`, and
    the second is that cluster's correspondence.
    '''

    if not clusters:
        return None

    best_clusters = [(-1, set())] * cover
    for i, (mean, resources) in enumerate(clusters):
        in_cluster = [resource.uri for resource in resources]
        correspondence = set(resources_to_cover) & set(in_cluster)

        if len(correspondence) >= len(best_clusters[0][1]):
            best_clusters[0] = (i, correspondence)
            best_clusters.sort(key = lambda e: (len(e[1]), e[0] != -1))

    return best_clusters

def predict_for_page_load(page, hindex):
    host = hindex[page.host]

    # cover resources from last load with clusters
    clusters, subclusters = host.clusters, host.subclusters
    res_from_last_load = page.get_resources_from_last_load()
    if not res_from_last_load:
        print '{} has no resources from last load!'.format(page.uri)
        return None, None

    cover_clusters = pick_best_cover(
        res_from_last_load, clusters, CLUSTER_COVER_SIZE)
    if cover_clusters is None:
        print 'No clusters for host ' + host.name
        return None, None

    print 'Cover clusters: {}'.format([i for i, _ in cover_clusters])

    # compute the list of all subclusters in all of the best clusters,
    # and all the resources that have been covered
    to_cover_subclusters = []
    covered_resources = set()
    for cluster, correspondence in cover_clusters:
        to_cover_subclusters += subclusters[cluster]
        covered_resources = covered_resources.union(correspondence)

    cover_subclusters = pick_best_cover(covered_resources,
        to_cover_subclusters, SUBCLUSTER_COVER_SIZE)
    print 'Cover subclusters: {}, sizes {}'.format(
        [i for i, _ in cover_subclusters],
        [len(to_cover_subclusters[i][1]) for i, _ in cover_subclusters]
    )

    # pick some subclusters to grow our set of predicted resources,
    # but never more than twice the size of the covered resources
    predicted = set(res_from_last_load)
    for i, correspondence in cover_subclusters:
        ruris = [resource.uri for resource in to_cover_subclusters[i][1]]
        with_subcluster = predicted.union(ruris)

        if len(with_subcluster) < 1.5 * len(res_from_last_load):
            predicted = predicted.union(ruris)
            print 'Picking {} (size {})'.format(i, len(predicted))

    cover_clusters = tuple(idx for idx, _ in cover_clusters)
    return predicted, cover_clusters

def predict_for_unknown_page(host):
    clusters = find_most_important_clusters(host.clusters, CLUSTER_COVER_SIZE)
    if not clusters:
        print 'No clusters for host ' + host.name
        return None, None

    subclusters = []
    for cidx, _ in clusters:
        subclusters += host.subclusters[cidx]

    predicted = []
    for sidx, _ in find_most_important_clusters(subclusters,
            SUBCLUSTER_COVER_SIZE):
        mean, resources = subclusters[sidx]
        predicted += [r.uri for r in resources]

    return predicted, [c[0] for c in clusters]

def visualize(host, chosen_clusters, predicted, explicit):
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
        if cidx in chosen_clusters:
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

        color = 'red' if cidx in chosen_clusters else colors.pop(0)

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
    plot.title('Clusters for host (red are the selected clusters)')
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

def simulate_predict_for_page_load(page_uri):
    with sqlite3.connect(dbfile) as db:
        hindex, pindex, rindex = load_database(db)

    huri = get_host_for_uri(page_uri)
    host = hindex[huri]
    cluster_resources_for_host(host, rindex)

    page = pindex.get(page_uri)
    if page is None:
        predicted, clusters = predict_for_unknown_page(host)
        if predicted is None:
            print 'No prediction.'
            return

        explicit = []
        print 'Would take predictive actions for {0} resources' \
               .format(len(predicted))
    else:
        explicit = page.get_resources_from_last_load()
        predicted, clusters = predict_for_page_load(page, hindex)
        if predicted is None:
            print 'No prediction.'
            return

        explicit_predicted = set(predicted) & set(explicit)
        print 'Would take predictive actions for {0} resources, ' \
              'out of which {1} were explicitly loaded last time, ' \
              'and the page loaded a total of {2} resources last time ' \
              '(so {3:.2f}% of explicit resources were predicted)' \
              .format(len(predicted),
                      len(explicit_predicted),
                      len(explicit),
                      (100.0 * len(explicit_predicted)) / len(explicit))

    visualize(host, clusters, predicted, explicit)

def cluster(dbfile):
    global NOW

    NOW = datetime.utcnow()
    with sqlite3.connect(dbfile) as db:
        hindex, pindex, rindex = load_database(db)

        for host in hindex.values():
            print 'Clustering ' + host.name
            cluster_resources_for_host(host, rindex)

        cursor = db.cursor()
        cursor.execute(
            'create table if not exists ' +
            'moz_page_predictions(pid integer, ruri text not null, ' +
            'foreign key(pid) references moz_pages(id))')
        cursor.execute(
            'create table if not exists ' +
            'moz_host_predictions(hid integer, ruri text not null, ' +
            'foreign key(hid) references moz_pages(id))')
        cursor.execute('delete from moz_page_predictions')
        cursor.execute('delete from moz_host_predictions')

        for page in pindex.values():
            predicted, _ = predict_for_page_load(page, hindex)
            if predicted is None:
                continue

            for ruri in predicted:
                record = (page.id, ruri)
                cursor.execute(
                    'insert into moz_page_predictions values (?, ?)', record)

        for host in hindex.values():
            predicted, _ = predict_for_unknown_page(host)
            if predicted is None:
                continue

            for ruri in predicted:
                record = (host.id, ruri)
                cursor.execute(
                    'insert into moz_host_predictions values (?, ?)', record)

def watch(dbfile):
    global NOW

    while True:
        cluster(dbfile)
        time.sleep(SLEEP_TIME_SECONDS)

if __name__ == "__main__":
    import sys

    action = sys.argv[1]
    dbfile = sys.argv[2]

    if action == 'predict':
        puri = sys.argv[3]
        simulate_predict_for_page_load(puri)
    elif action == 'watch':
        watch(dbfile)
    elif action == 'cluster':
        cluster(dbfile)
    else:
        print >> sys.stderr, 'Invalid action: use predict or watch'
        sys.exit(1)

