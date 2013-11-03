import sys
import json
import padnums
import sqlite3

def get_keys(d, *keys):
    return reduce(dict.get, keys, d)

def make_table_for_metric(metric, kmeansf, seerf, noseerf):
    rows = []
    names_and_result_files = zip(
        ("kmeans", "seer", "noseer"),
        (kmeansf, seerf, noseerf)
    )

    for name, resultf in names_and_result_files:
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

if __name__ == "__main__":
    dbfile = sys.argv[1]
    kmeansf, seerf, noseerf = sys.argv[2:]

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

    requests_predicted = set(prediction) & set(all_requests)
    correct_predictions_ratio = len(requests_predicted) / float(len(prediction))
    predicted_requests_ratio = len(requests_predicted) / float(len(all_requests))

    print "At least %f predictions were correct" % correct_predictions_ratio
    print "At least %f requests were predicted" % predicted_requests_ratio
    print

    speedindex_table = make_table_for_metric("SpeedIndex", kmeansf, seerf, noseerf)
    print "SpeedIndex"
    padnums.pprint_table(sys.stdout, speedindex_table)
