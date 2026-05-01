import sqlite3
c = sqlite3.connect('conceptnet.db')
cur = c.cursor()
cur.execute("SELECT COUNT(*) FROM concept")
print("Total concepts:", cur.fetchone()[0])
cur.execute("SELECT id FROM language WHERE name = 'en'")
en_id = cur.fetchone()[0]
print("English language ID:", en_id)
cur.execute("SELECT COUNT(*) FROM label WHERE language_id = ?", (en_id,))
print("English labels:", cur.fetchone()[0])