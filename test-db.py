import sqlite3

conn = sqlite3.connect("brns-db")

result = conn.execute("SELECT * FROM mode")

print(list(result)[0][1])
