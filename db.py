import sqlite3

def get_connection():
    conn = sqlite3.connect("brns-db")
    return conn

def update_mode(mode):
    conn = sqlite3.connect("brns-db")
    conn.execute(f"UPDATE mode SET current_mode='{mode}' WHERE id=1")
    conn.commit()
    conn.close()
