import sqlite3

conn = sqlite3.connect("agent_memory.db")
cursor = conn.cursor()

cursor.execute("""
    SELECT convention_text, category, times_confirmed, source_pr_number 
    FROM conventions 
    WHERE active = 1 
    ORDER BY times_confirmed DESC, id ASC
""")

rows = cursor.fetchall()
print(f"Total active conventions: {len(rows)}\n")

print("=== Reinforced conventions (times_confirmed > 1) ===")
reinforced_count = 0
for text, category, count, pr in rows:
    if count > 1:
        print(f"[{count}x] ({category}) {text}")
        reinforced_count += 1

print(f"\n({reinforced_count} conventions reinforced 2+ times, out of {len(rows)} total)")

conn.close()