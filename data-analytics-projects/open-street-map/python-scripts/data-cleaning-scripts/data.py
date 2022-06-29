import csv
import codecs
import pprint
import re
import xml.etree.cElementTree as ET

import cerberus

import schema

OSM_PATH = "Honolulu.osm"

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

SCHEMA = schema.schema

# Make sure the fields order in the csvs matches the column order in the sql table schema
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']

street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
city_type_re = re.compile(r'\w+.?\w+', re.IGNORECASE)

expected = ["Street","Circle", "Avenue","Way", "West", "North","Highway", "East", "South",
            "Boulevard", "Drive", "Court", "Place", "Square", "Koolau", "Lane", 
            "Road", "Trail", "Parkway", "Commons", "Center", "Mall", "Terrace", "Walk", "Loop"]

expected_cities = ["Honolulu", "Aiea", "Kailua", "Kaneohe", "Pearl City", "Fort Shafter", "Waipahu", "Waimanalo", "Ewa Beach"]

mapping = { "St": "Street", "St.": "Street", "Rd": "Road", "Rd.": "Road", "Ave": "Avenue", "Blvd": "Boulevard", "Blvd.": "Boulevard", "Dr": "Drive", "Hwy.": "Highway", "street": "Street", "Pl": "Place" }

city_mapping = { "Honlulu": "Honolulu", "Honollulu": "Honolulu", "Kailuna": "Kailua", "honolulu": "Honolulu", "waimanalo": "Waimanalo"}

def process_key(key_string): 
      """
      This function processes 'k' values to slice and separate key strings into
      their respective keys and tag types. It returns an ordered listed with
      the new key and the tag type. 
      
      Args:
      
           key_string (str): A tags 'k' attribute value, is the key_string.
           
      Returns:
      
           if ":" is found inside Args:
           
            tag_type (str): characters in key_string before ":"
            new_key  (str): characters in key_string after ":"
           
           if no ":" is found inside Args:
            
            tag_type (str): "regular"
            new_key (str): key_string
            
      """
      if ":" in key_string:   
            indexed_string = key_string.find(":")
            tag_type = key_string[:indexed_string]
            new_key = key_string[indexed_string+1:]
            return [new_key, tag_type]
      else:
            new_key = key_string
            tag_type = "regular"
            return [new_key, tag_type]
            
def update_name(name, mapping):

    
    m = street_type_re.search(name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            name = re.sub(street_type_re, mapping[street_type], name)

    return name
def update_city(name, city_mapping):

    m = city_type_re.search(name)
    if m:
        city_type = m.group()
        if city_type not in expected_cities:
            name = re.sub(city_type_re, city_mapping[city_type], name)

    return name

def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []  

    
    if element.tag == 'node':
        
        # first iterate over nodes to get attr. names and values into a dictionary (node_attribs)
        for attrib_name, attrib_value in element.attrib.items():
            if attrib_name in NODE_FIELDS:
                node_attribs[attrib_name] = attrib_value
        
        
        '''
        Next loop through the child tags for nodes and parse the
        key and value pair for each tag and to also clean the key data. 
        We then build a dictionary to hold the key and value pairs, so we can
        easily append the dictionary to the "tags" list
        '''
        
        for i in element.iter('tag'):
            
            
            node_dict = {}
            if PROBLEMCHARS.search(i.attrib['k']):
                continue
            elif LOWER_COLON.search(i.attrib['k']):
                node_dict['id'] = element.attrib['id']
                node_dict['key'] = process_key(i.attrib['k'])[0]
                node_dict['type'] = process_key(i.attrib['k'])[1]
                
                #cleaning function
                if i.attrib['k'] == 'addr:street':
                    node_dict['value'] = update_name(i.attrib['v'], mapping)
                #cleaning city name function
                elif i.attrib['k'] == 'addr:city':
                    node_dict['value'] = update_city(i.attrib['v'], city_mapping)
                else:
                    node_dict['value'] = i.attrib['v']
            
            else:
                node_dict['id'] = element.attrib['id']
                node_dict['key'] = process_key(i.attrib['k'])[0]
                node_dict['type'] = process_key(i.attrib['k'])[1]
                node_dict['value'] = i.attrib['v']
                
                
                tags.append(node_dict)
       
       
        
        return {'node': node_attribs, 'node_tags': tags}
    
    elif element.tag == 'way':
        
        for attrib_name, attrib_value in element.attrib.items():
            if attrib_name in WAY_FIELDS:
                way_attribs[attrib_name] = attrib_value
        
        '''
        Now for the way tags, because they are the same setup as node tags
        we can do the same process as above
        '''
        
        
        for i in element.iter('tag'):
            
            
            
            way_dict = {}
            if PROBLEMCHARS.search(i.attrib['k']):
                continue
            elif LOWER_COLON.search(i.attrib['k']):
                way_dict['id'] = element.attrib['id']
                way_dict['key'] = process_key(i.attrib['k'])[0]
                way_dict['type'] = process_key(i.attrib['k'])[1]
                
                #cleaning function
                if i.attrib['k'] == 'addr:street':
                    way_dict['value'] = update_name(i.attrib['v'], mapping)
                #cleaning city name function
                elif i.attrib['k'] == 'addr:city':
                    way_dict['value'] = update_city(i.attrib['v'], city_mapping)
                
                else:
                    way_dict['value'] = i.attrib['v']
            else:
                way_dict['id'] = element.attrib['id']
                way_dict['key'] = process_key(i.attrib['k'])[0]
                way_dict['type'] = process_key(i.attrib['k'])[1]
                way_dict['value'] = i.attrib['v']
            tags.append(way_dict)
        print tags
        
        """
        enumerate() is used here to create a counter for each 'nd' child node.
        """

        for n, i in enumerate(element.iter('nd')):
            nd_dict = {}
            nd_dict['id'] = element.attrib['id']
            nd_dict['node_id'] = i.attrib['ref']
            nd_dict['position'] = n
            way_nodes.append(nd_dict)
        
                
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}


# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema"""
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_string = pprint.pformat(errors)
        
        raise Exception(message_string.format(field, error_string))


class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file, \
         codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file, \
         codecs.open(WAYS_PATH, 'w') as ways_file, \
         codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file, \
         codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])


process_map(OSM_PATH, validate=True)
    
