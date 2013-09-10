import time
import pprint
import sqlite3
from urllib2 import urlparse
from collections import defaultdict
from datetime import datetime

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
'''

NOW = time.time() * 1e6
MAX_CLUSTER_MEAN_DISTANCE = 1e-1

class Seer(object):
    def __init__(self):
        self.clusters_for_hosts = {} # host -> (mean, [elements])

        # for visualization/debugging only
        self.timestamp_and_hits_for_uris = {} # uri -> (timestamp, hits)

    def learn_subresource(self, page, uri, timestamp_us, hits):
        host = self.get_host_for_uri(page)
        self.learn_uri(host, uri, timestamp_us, hits)

    def learn_pageload(self, uri, timestamp_us, hits):
        host = self.get_host_for_uri(uri)
        self.learn_uri(host, uri, timestamp_us, hits)

    # TODO of course we don't really want to receive the hits and
    # timestamp all at once
    def learn_uri(self, host, uri, timestamp_us, hits):
        self.timestamp_and_hits_for_uris[uri] = (timestamp_us, hits)

        clusters = self.clusters_for_hosts.setdefault(host, [])
        vec = self.make_description_vector(timestamp_us, hits)

        closest_idx, d = self.find_closest_cluster(vec, clusters)
        if d > MAX_CLUSTER_MEAN_DISTANCE:
            # create a new cluster
            # TODO how many clusters do we want to maintain?
            clusters.append((vec, [uri]))
            return

        # add uri to known cluster
        closest = clusters[closest_idx]
        mean, elements = closest
        elements.append(uri)
        mean += (1.0 / len(elements)) * (vec - mean)

        self.clusters_for_hosts[host][closest_idx] = (mean, elements)

    # TODO we shouldn't receive timestamp and hits here, but load them
    # from db instead
    def predict_for_page_load(self, uri, timestamp_us, hits):
        host = self.get_host_for_uri(uri)
        if host not in self.clusters_for_hosts:
            return []

        clusters = self.clusters_for_hosts[host]
        vec = self.make_description_vector(timestamp_us, hits)
        closest_idx, d = self.find_closest_cluster(vec, clusters)

        if closest_idx is None:
            return []

        self.visualize(uri, clusters, closest_idx)
        return clusters[closest_idx][1]

    def get_host_for_uri(self, uri):
        parts = urlparse.urlparse(uri)
        return parts.scheme + '://' + parts.netloc

    def make_description_vector(self, timestamp_us, hits):
        return np.array((timestamp_us / NOW, hits))

        # dt = datetime.fromtimestamp(timestamp_us / 1e6)
        # return np.array((dt.month, dt.day, dt.hour, dt.minute, dt.second, hits))

    def find_closest_cluster(self, vec, clusters):
        closest = None
        mindist = float('inf')
        for i, (mean, _) in enumerate(clusters):
            d = np.linalg.norm(mean - vec)
            if d < mindist:
                closest, mindist = i, d

        return closest, mindist

    def visualize(self, page, clusters, closest_cluster):
        colors = [
            'black',
            'yellow',
            'green',
            'blue',
            'orange',
            'gray'
        ]

        page_coords = self.make_description_vector(
            *self.timestamp_and_hits_for_uris[page]
        )

        print 'Page is at: {0}'.format(page_coords)
        plot.plot(page_coords[0], page_coords[1], color = 'brown', marker = '^')
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

            print '{0} cluster is at {1} with {2} elements'.format(
                    color, mean, len(sres))

            plot.plot(*mean, color = color, marker = 'v')
            for sr in sres:
                sres_vec = self.make_description_vector(
                    *self.timestamp_and_hits_for_uris[sr]
                )

                plot.plot(sres_vec[0], sres_vec[1], color = color, marker = 'o')

        plot.title('Clusters for host')
        plot.ylabel('hits')
        plot.xlabel('timestamp')

        plot.show()

def load_db(dbfile):
    db = sqlite3.connect(dbfile)
    cursor = db.cursor()

    cursor.execute('SELECT * FROM moz_subresources;')
    subresources = cursor.fetchall()

    cursor.execute('SELECT * FROM moz_pages;')
    pages = cursor.fetchall()

    db.close()
    return pages, subresources

def find_page_by_id(id, pages):
    for p in pages:
        if p[0] == id:
            return p

    assert False

if __name__ == "__main__":
    import sys

    seer = Seer()
    pages, subresources = load_db(sys.argv[1])

    visited_page = None
    for page in pages:
        id, uri, hits, timestamp_us = page
        seer.learn_pageload(uri, timestamp_us, hits)

        if uri == sys.argv[2]:
            visited_page = page

    for sres in subresources:
        id, pid, uri, hits, timestamp_us = sres
        page = find_page_by_id(pid, pages)
        seer.learn_subresource(page[1], uri, timestamp_us, hits)

    if visited_page is None:
        print >>sys.stderr, 'Asked predict an unknown page!'

    explicit = [sr for sr in subresources if sr[1] == visited_page[0]]
    predicted = seer.predict_for_page_load(visited_page[1], visited_page[3], visited_page[2])

    print 'Would take predictive actions for {0} items, ' \
          'out of which {1} were explicitly loaded last time, ' \
          'and the page loaded a total of {2} subresources' \
          .format(len(predicted), len(set(predicted) & set(explicit)), len(explicit))
