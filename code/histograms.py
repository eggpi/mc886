import sys
import sqlite3
import matplotlib.pyplot as plot

db = sqlite3.connect(sys.argv[1])
cursor = db.cursor()

cursor.execute('SELECT COUNT(*) FROM moz_subresources GROUP BY pid;')
counts = cursor.fetchall()

plot.subplot(311)
plot.hist([c[0] for c in counts], bins = 100, range = (0, 50))
plot.title('Subresources per page')

cursor.execute('SELECT max(last_hit) FROM moz_subresources group by uri;')
last_loads = cursor.fetchall()

plot.subplot(312)
plot.hist([l[0] for l in last_loads], bins = 100)
plot.title('Timestamp per subresource')

cursor.execute('SELECT sum(hits) FROM moz_subresources group by uri;')
hits = cursor.fetchall()

plot.subplot(313)
plot.hist([h[0] for h in hits], bins = 100)
plot.title('Hits per subresource')

plot.show()
