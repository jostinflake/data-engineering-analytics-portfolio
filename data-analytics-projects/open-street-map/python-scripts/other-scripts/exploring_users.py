#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET
import pprint
import re

def get_user(element):
    return

def process_map(filename):
    users = set()
    for _, element in ET.iterparse(filename):
      if "uid" in element.attrib:
          users.add(element.attrib["uid"])
    return users

users = process_map("HonoluluSample.osm")

pprint.pprint(len(users))