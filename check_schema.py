import sqlite3
c = sqlite3.connect('conceptnet.db')
cur = c.cursor()
cur.execute("PRAGMA table_info(language)")
print("language columns:", cur.fetchall())
cur.execute("SELECT * FROM language LIMIT 5")
print("language sample:", cur.fetchall())