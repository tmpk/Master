# Script for extracting desired data from AIS records in geojson format, and writing to file.
# The output file stores individual sailings as one LineString, so that there is a one-to-one
# correspondence between sailings and LineStrings.

import geojson
import numpy as np
import json
import dateutil.parser
import datetime
import pandas as pd


with open('skipshastighet_2016/all/tracks.geojson') as f1:
    data1 = geojson.load(f1)

with open('skipshastighet_2017/all/tracks.geojson') as f2:
     data2 = geojson.load(f2)
    
with open('skipshastighet_2018/all/tracks.geojson') as f3:
     data3 = geojson.load(f3)
    
with open('skipshastighet_2019/all/tracks.geojson') as f4:
     data4 = geojson.load(f4)

with open('skipshastighet_2020_jan-mai/ravnkloa/tracks.geojson') as f5:
     data5 = geojson.load(f5)
 

def df_to_geojson(df, properties):
    # Function for converting a dataframe into geojson with desired format

    geojson = {'type':'FeatureCollection', 'features':[]}
    for _, row in df.iterrows():
        feature = {'type':'Feature',
                   'properties':{},
                   'geometry':{'type':[],
                               'coordinates':[]}}
        feature['geometry']['type'] = 'LineString'
        feature['geometry']['coordinates'] = row['coords']
        for prop in properties:
            feature['properties'][prop] = row[prop]
        geojson['features'].append(feature)

    return geojson

def chain(lists):
    # Recursive function used for chaining together a sailing represented as a 
    # MultiLineString. 
    # Made redundant by later server-side improvements to data format,
    # which caused all sailings to be represented as LineStrings. 

    if len(lists) <= 1:
        return lists
    elif len(lists) == 2:
        if lists[0][-1] == lists[1][0]:
            v1=lists[0][0:-1]
            v2=lists[1]
            v1.extend(v2)
            return [v1]
        else:
            return [lists[0], lists[1]]
    else:
        if lists[0][-1] == lists[1][0]:
            v1=lists[0][0:-1]
            v2=lists[1]
            v1.extend(v2)
            v3 = lists[2:]
            v4 = []
            for c in v3:
                v4.extend(c)
            new_list = [v1, v4]
            return chain(new_list)
        else:
            return chain(lists[1:])

out = []
row = [None, None, None, None, [0, 0], None]
for data in [data1, data2, data3, data4, data5]:
    for feature in data['features']:
        mmsi = feature['properties']['mmsi']
        start = feature['properties']['start']
        end = feature['properties']['end'] 
        endtime = feature['properties']['endt']
        coords = feature['geometry']['coordinates']
        speed = feature['properties']['speed']

        ### leave out data identified as spurious:
        if mmsi == 259171000 and speed == 102.3:
            continue
        if mmsi == 257465900 or mmsi == 257234800: 
            continue
        ###

        if start == row[2] and mmsi == row[0] and coords[0] == row[4][-1]:
            new_coords = row[4]
            new_coords.append(coords[1])
            if isinstance(row[5], list):
                speeds = row[5]
                speeds.append(speed)
            else:
                speeds = [row[5], speed]
            row = [mmsi, row[1], end, endtime, new_coords, speeds]
            del out[-1]
        else:
            row = [mmsi, start, end, endtime, coords, speed]

        out.append(row)

df = pd.DataFrame(out)

df.rename(columns={0: 'MMSI#', 1: 'start', 2: 'end', 3: 'end time', 4: 'coords', 5: 'speed'}, inplace=True)
properties = ['MMSI#', 'start', 'end', 'end time', 'speed']

gj = df_to_geojson(df, properties)

geojson_str = geojson.dumps(gj, indent=2)
output_filename = 'dataset_2016-2020_chained.geojson'
with open(output_filename, 'w') as output_file:
    output_file.write('{}'.format(geojson_str))