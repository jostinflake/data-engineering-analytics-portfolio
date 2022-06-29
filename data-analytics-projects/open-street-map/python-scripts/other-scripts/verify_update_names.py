import sqlite3

problematic_abbreviations = ['Ave', 'Blvd', 'Blvd.', 'Dr', 'Hwy.', 'St', 'St.', 'street']

# Fetch the street names for way tags.
db = sqlite3.connect("openstreetmap_honolulu.db")
c = db.cursor()
query = "SELECT value FROM ways_tags WHERE key = 'street';"
c.execute(query)
rows = c.fetchall()

street_type_list = []

for row in rows:
    new_list = row[0].split(' ')
    street_type_list.append(new_list[-1])
    
print "Incorrect Street names that still exist:"

num_bad_street = 0

for street_type in street_type_list:
    if street_type in problematic_abbreviations:
        num_bad_street += 1
        print street_type
    else:
        continue

if num_bad_street == 0:
    print(" ")
    print("No incorrect street names exist!")
else:
    pass
        
db.close()