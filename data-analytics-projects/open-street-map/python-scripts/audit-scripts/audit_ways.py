import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

osm_file = open("Honolulu.osm", "r")

street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
street_types = defaultdict(set)

expected = ["Street","Circle", "Avenue","Way",
            "West", "North","Highway", "East", "South", 
            "Boulevard", "Drive", "Court", "Place", "Koolau", 
            "Square", "Lane", "Road", "Trail", "Parkway",
             "Commons", "Center", "Mall", "Terrace", "Walk",
             "Loop"]

def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)

def print_sorted_dict(d):
    keys = d.keys
    keys = sorted(keys, key=lambda s: s.lower())
    for k in keys:
        v = d[k]
        print "%s: %d" % (k, v)
        
def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

def audit_ways():
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    pprint.pprint(dict(street_types))
#    print_sorted_dict(street_types)

audit_ways()