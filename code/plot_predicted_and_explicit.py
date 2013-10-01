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
        clusters_for_hosts, subclusters_for_clusters, distortions = \
            moz_kmeans.cluster_subresources_for_hosts(moz_kmeans.K, db)

        cursor = db.cursor()
        cursor.execute('select uri from moz_pages;')
        pages = filter(pages_re.match, (row[0] for row in cursor))

        for page_uri in pages:
            cursor = db.cursor()
            cursor.execute('select * from moz_pages where uri = ?', (page_uri,))
            page = cursor.fetchone()

            result = moz_kmeans.predict_for_page_load(
                db, page, clusters_for_hosts, subclusters_for_clusters)

            if result is not None:
                _, _, predicted_sres, explicit_sres = result
            else:
                continue

            plot.subplot(221)
            plot.plot(len(explicit_sres), len(predicted_sres), 'bo')

            explicit_predicted_over_predicted.append(
                float(len(predicted_sres & explicit_sres)) / len(predicted_sres))
            explicit_predicted_over_explicit.append(
                float(len(predicted_sres & explicit_sres)) / len(explicit_sres))

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
