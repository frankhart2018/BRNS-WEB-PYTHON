import sqlite3

conn = sqlite3.connect("brns-db")

result = conn.execute("SELECT * FROM mode")

print(list(result)[0][1])


from db import update_scan_count

update_scan_count()
