import os
import sys
import json
import padnums
import sqlite3
from urllib2 import urlparse

def get_host_for_uri(uri):
    parts = urlparse.urlparse(uri)
    return parts.scheme + '://' + parts.netloc

def get_keys(d, *keys):
    return reduce(dict.get, keys, d)

def make_table_for_metric(metric, kmeansf, seerf, noseerf):
    rows = []
    names_and_result_files = zip(
        ("kmeans", "seer", "noseer"),
        (kmeansf, seerf, noseerf)
    )

    for name, resultf in names_and_result_files:
        if not os.path.isfile(resultf):
            continue

        with open(resultf) as f:
            data = json.load(f)["data"]
            rows.append([
                name,
                get_keys(data, "median", "firstView", metric),
                get_keys(data, "average", "firstView", metric),
                get_keys(data, "standardDeviation", "firstView", metric),
                get_keys(data, "median", "repeatView", metric),
                get_keys(data, "average", "repeatView", metric),
                get_keys(data, "standardDeviation", "repeatView", metric),
                get_keys(data, "summary")
            ])

    headers = [
        "", "first view median", "first view avg", "first view stdev",
        "repeated view median", "repeated view avg", "repeated view stdev",
        "summary"
    ]

    return [headers] + rows

def summarize_experiment(dbfile, kmeansf, seerf, noseerf):
    kmeans_results = json.load(open(kmeansf))
    data = kmeans_results["data"]

    all_requests = []
    for req in get_keys(data, "runs", "1", "firstView", "requests"):
        all_requests.append(req["full_url"])

    with sqlite3.connect(dbfile) as db:
        cursor = db.cursor()

        cursor.execute(
            "select ruri from moz_page_predictions, moz_pages "
            "where moz_pages.id = moz_page_predictions.pid and "
            "moz_pages.uri = ?", (data["testUrl"],))

        prediction = [tp[0] for tp in cursor]

        if not prediction:
            host = get_host_for_uri(data["testUrl"])

            cursor.execute(
                "select ruri from moz_host_predictions, moz_hosts "
                "where moz_hosts.id = moz_host_predictions.hid and "
                "moz_hosts.origin = ?", (host,))

            prediction = [tp[0] for tp in cursor]

        cursor.execute(
            "select moz_subresources.uri from moz_subresources, moz_pages "
            "where moz_pages.id = moz_subresources.pid and "
            "moz_pages.uri = ?", (data["testUrl"],))

        known_resources = [tp[0] for tp in cursor]

    requests_predicted = set(prediction) & set(all_requests)
    known_predicted = set(known_resources) & set(prediction)

    correct_predictions_ratio = 100 * len(requests_predicted) / float(len(prediction))
    predicted_requests_ratio = 100 * len(requests_predicted) / float(len(all_requests))
    known_requests_ratio = 100 * len(known_resources) / float(len(all_requests))

    print "At least %.2f %% predictions were correct" % correct_predictions_ratio
    print "At least %.2f %% requests were predicted" % predicted_requests_ratio
    print "At least %.2f %% requests were known" % known_requests_ratio
    if known_resources:
        known_requests_predicted_ratio = 100 * len(known_predicted) / float(len(known_resources))
        print "Out of those, %.2f %% were predicted" % known_requests_predicted_ratio
    print

    speedindex_table = make_table_for_metric("SpeedIndex", kmeansf, seerf, noseerf)
    print "SpeedIndex"
    padnums.pprint_table(sys.stdout, speedindex_table)

if __name__ == "__main__":
    expdir = sys.argv[1]

    dbfile = os.path.join(expdir, "seer.sqlite")
    for expname in os.listdir(expdir):
        exppath = os.path.join(expdir, expname)
        if os.path.isdir(exppath):
            kmeansf = os.path.join(exppath, "kmeans.results.json")
            seerf = os.path.join(exppath, "seer.results.json")
            noseerf = os.path.join(exppath, "noseer.results.json")

            print "Experiment: " + expname
            summarize_experiment(dbfile, kmeansf, seerf, noseerf)
            print "-------------"
