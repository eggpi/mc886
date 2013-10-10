import re
import sqlite3
import matplotlib.pyplot as plot

import moz_kmeans

if __name__ == "__main__":
    import sys

    dbfile = sys.argv[1]
    pages_re = re.compile(sys.argv[2])

    explicit_predicted_over_explicit = []
    explicit_predicted_over_predicted = []
    with sqlite3.connect(dbfile) as db:
        hindex, pindex, rindex = moz_kmeans.load_database(db)

        cursor = db.cursor()
        cursor.execute('select uri from moz_pages;')
        pages = filter(pages_re.match, (row[0] for row in cursor))

        for page_uri in pages:
            page = pindex[page_uri]
            host = hindex[page.host]

            if not host.clusters:
                moz_kmeans.cluster_resources_for_host(host, rindex)

            result = moz_kmeans.predict_for_page_load(page, hindex)
            if result is None:
                continue

            cidx, predicted = result
            predicted = set(predicted)
            explicit = set(page.get_resources_from_last_load())

            plot.subplot(221)
            plot.plot(len(explicit), len(predicted), 'bo')

            explicit_predicted_over_predicted.append(
                float(len(predicted & explicit)) / len(predicted))
            explicit_predicted_over_explicit.append(
                float(len(predicted & explicit)) / len(explicit))

    plot.subplot(221)
    plot.xlabel('explicitly loaded subresources')
    plot.ylabel('predicted subresources')

    plot.subplot(222)
    plot.hist(explicit_predicted_over_predicted)
    plot.title('explicit predicted / predicted')

    plot.subplot(223)
    plot.hist(explicit_predicted_over_explicit)
    plot.title('explicit predicted / explicit')

    plot.show()
