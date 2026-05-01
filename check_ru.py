import sqlite3
c = sqlite3.connect('conceptnet.db')
cur = c.cursor()
cur.execute("SELECT id FROM language WHERE name = 'ru'")
ru_id = cur.fetchone()
print("Russian lang ID:", ru_id)
cur.execute("SELECT COUNT(*) FROM label WHERE language_id = ?", (ru_id[0],))
print("Russian labels:", cur.fetchone()[0])