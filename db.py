import sqlite3

def get_connection():
    conn = sqlite3.connect("brns-db")
    return conn

def update_mode(mode):
    conn = get_connection()
    conn.execute(f"UPDATE mode SET current_mode='{mode}' WHERE id=1")
    conn.commit()
    conn.close()

def update_scan_count():
    conn = get_connection()
    cursor = conn.execute(f"SELECT * FROM count WHERE id=1")
    current_count = int(list(cursor)[0][1])
    conn.execute(f"UPDATE count SET cnt={current_count + 1} WHERE id=1")
    conn.commit()
    conn.close()

    return current_count + 1
