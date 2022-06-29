import xml.etree.cElementTree as ET
import pprint

filename = "HonoluluSample.osm"

def count_tags(filename):
       tags = {}
       for event, elem in ET.iterparse(filename):
           if elem.tag not in tags:
               tags[elem.tag] = 1
           else:
               tags[elem.tag] += 1
       return tags

pprint.pprint(count_tags(filename))