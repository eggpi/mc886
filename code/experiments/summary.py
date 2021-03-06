import os
import glob
import sys
import json
import padnums
import sqlite3
import itertools
from urllib2 import urlparse

def get_host_for_uri(uri):
    parts = urlparse.urlparse(uri)
    return parts.scheme + '://' + parts.netloc

def get_keys(d, *keys):
    return reduce(dict.get, keys, d)

def make_table_for_metric(metric, resultsd):
    rows = []

    names_and_result_files = []
    for resultsf in glob.glob(os.path.join(resultsd, "*.results.json")):
        names_and_result_files.append((
            resultsf.replace(".results.json", ""), resultsf))

    fv_measures = {}
    connectivity = []
    for name, resultf in names_and_result_files:
        if not os.path.isfile(resultf):
            continue

        with open(resultf) as f:
            data = json.load(f)["data"]
            rows.append([
                name,
                float(get_keys(data, "median", "firstView", metric)),
                float(get_keys(data, "average", "firstView", metric)),
                float(get_keys(data, "standardDeviation", "firstView", metric)),
                float(get_keys(data, "median", "repeatView", metric)),
                float(get_keys(data, "average", "repeatView", metric)),
                float(get_keys(data, "standardDeviation", "repeatView", metric)),
                get_keys(data, "summary"),
                get_keys(data, "successfulFVRuns"),
                get_keys(data, "successfulRVRuns"),
                get_keys(data, "connectivity")
            ])

            connectivity.append(get_keys(data, "connectivity"))

            fv_measures[name] = []
            runs = int(get_keys(data, "successfulFVRuns"))
            for i in range(1, runs + 1):
                fv_measures[name].append(
                    get_keys(data, "runs", str(i), "firstView", metric))

    assert len(set(connectivity)) == 1

    friedman(fv_measures)

    headers = [
        "", "fv median", "fv avg", "fv stdev",
        "rv median", "rv avg", "rv stdev",
        "summary", "fv runs", "rv runs", "network"
    ]

    return [headers] + rows

def friedman(measures):
    k = len(measures.keys())
    assert k == 3

    n = len(measures.values()[0])
    assert len(set(map(len, measures.values()))) == 1
    assert n == 20

    # rank and add up each row
    labels = measures.keys()
    data = [measures[l] for l in labels]
    rowsums = dict((l, 0) for l in labels)
    for row in zip(*data):
        srow = sorted(row)

        i = 0
        while i < len(srow):
            m = srow[i]
            for j in range(i, len(srow)):
                if srow[j] != m:
                    break
            else:
                j += 1

            if i == j - 1:
                rowsums[labels[row.index(m)]] += i + 1
            else:
                rank = float(sum(range(i + 1, j - i + 1))) / (j - i)
                idx = -1
                for _ in range(j - i):
                    idx = row.index(m, idx + 1)
                    rowsums[labels[idx]] += rank

                try:
                    row.index(m, idx + 1) == -1
                except ValueError:
                    pass
                else:
                    assert False

            i = j

    sqrowsums = dict((l, s ** 2) for l, s in rowsums.items())
    M = 12.0 / (n * k * (k + 1)) * sum(sqrowsums.values()) - 3 * n * (k + 1)

    if M > 6.3:
        print "The difference IS statistically significant"
    else:
        print "The difference IS NOT statistically significant"
        return

    avgranks = dict((l, float(s) / n) for l, s in rowsums.items())
    critval = 2.343 * (k * (k + 1) / (6.0 * n))**0.5

    for a, b in itertools.combinations(labels, 2):
        diff = avgranks[a] - avgranks[b]
        if abs(diff) > critval:
            print "%s > %s" % ((b, a) if diff > 0 else (a, b))

def summarize_experiment(dbfile, resultsd):
    try:
        kmeansf = os.path.join(resultsd, "kmeans.results.json")
        kmeans_results = json.load(open(kmeansf))
        data = kmeans_results["data"]

        all_requests = []
        for req in get_keys(data, "runs", "1", "firstView", "requests"):
            all_requests.append(req["full_url"])
        all_requests = set(all_requests)

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

        requests_predicted = [r for r in prediction if r in all_requests]
        known_predicted = [r for r in known_resources if r in prediction]
        known_requested = [r for r in all_requests if r in known_resources]

        correct_predictions_ratio = 100.0 * len(requests_predicted) / len(prediction)
        predicted_requests_ratio = 100.0 * len(requests_predicted) / len(all_requests)
        known_requests_ratio = 100.0 * len(known_requested) / len(all_requests)

        print "At least %.2f %% predictions were correct" % correct_predictions_ratio
        print "At least %.2f %% requests were predicted" % predicted_requests_ratio
        print "At least %.2f %% requests were known" % known_requests_ratio
        if known_resources:
            known_requests_predicted_ratio = 100 * len(known_predicted) / float(len(known_resources))
            print "Out of those, %.2f %% were predicted" % known_requests_predicted_ratio
        print
    except Exception as x:
        print "Failed to load kmeans statistics: " + str(x)

    speedindex_table = make_table_for_metric("SpeedIndex", resultsd)
    print "SpeedIndex"
    padnums.pprint_table(sys.stdout, speedindex_table)

if __name__ == "__main__":
    expdir = sys.argv[1]

    dbfile = os.path.join(expdir, "seer.sqlite")
    for expname in os.listdir(expdir):
        exppath = os.path.join(expdir, expname)
        if os.path.isdir(exppath):
            print "Experiment: " + expname
            summarize_experiment(dbfile, exppath)
            print "-------------"
