# Script for processing the AIS data. The area considered is a 1500 m x 700 m
# rectangle overlain with a square grid. Based on start- and endpoints defined 
# in each geojson LineString, ship trajectory is linearly interpolated and position
# is subsequently mapped to the appropriate cell in the grid. Moreover, the speed
# and heading of the vessel in each cell is recorded.  Sailings are also grouped as north-, south-, east- or westbound. 
# Measures to avoid recording positions which are on land have been taken.

import geojson
import numpy as np
import json
import dateutil.parser
import datetime
import pandas as pd 
import utm
import pymap3d as pm 

with open('dataset_2016-2020_chained.geojson') as f:
    data = geojson.load(f)

step = 5
x_bin = np.arange(0, 1500, step)
y_bin = np.arange(0, 700, step)
sect = np.pi/32
heading_bin1 = [0]
heading_bin1 = [i*sect for i in np.arange(1,32,2)]
heading_bin1.append(np.pi)
heading_bin2 = [-1*x for x in heading_bin1]
heading_bin2.reverse()
headings = ['N', 'NbE','NNE', 'NEbN', 'NE', 'NEbE', 'ENE', 'EbN', 'E', 'EbS', 'ESE', 'SEbE', 'SE', 'SEbS', 'SSE', 'SbE', 'S', 'SbW', 'SSW', 'SWbS', 'SW', 'SWbW', 'WSW', 'WbS', 'W', 'WbN', 'WNW', 'NWbW', 'NW', 'NWbN', 'NNW','NbW']
#   bin1     1    2      3      4      5      6       7      8     9     10    11     12      13    14      15     16     17     
#   bin2                                                                                                                  1     2     3      4     5     6     7      8    9     10   11      12     13     14    15       16    17  

speed_bin = [0, 2, 4, 6, 8, 10] # reduced speed bin

frequency_table = np.empty((int(700/step), int(1500/step)), dtype=object)
frequency_table_eb = np.empty((int(700/step), int(1500/step)), dtype=object)
frequency_table_wb = np.empty((int(700/step), int(1500/step)), dtype=object)
frequency_table_nb = np.empty((int(700/step), int(1500/step)), dtype=object)
frequency_table_sb = np.empty((int(700/step), int(1500/step)), dtype=object)

f_table_pos = np.zeros((int(700/step), int(1500/step)), dtype=int)
f_table_pos_nb = np.zeros((int(700/step), int(1500/step)), dtype=int)
f_table_pos_sb = np.zeros((int(700/step), int(1500/step)), dtype=int)
f_table_pos_eb = np.zeros((int(700/step), int(1500/step)), dtype=int)
f_table_pos_wb = np.zeros((int(700/step), int(1500/step)), dtype=int)

p0 = (63.43095, 10.37723, 0) # define origin of local ned system

def calc_coeffs(start, stop):
    # returns coefficients a, b for the linear interpolated line y = ax + b between "start", "stop";  points are in NED-frame
    a = (stop[0] - start[0]) / (stop[1] -start[1])
    b = start[0] - a*start[1]
    return a, b

# define line y = a_1*x + b_1 above which points are ignored
stop_ned_1 = pm.geodetic2ned(*(63.43592, 10.39167, 0), *p0)  
start_ned_1 = pm.geodetic2ned(*(63.43247, 10.38001, 0), *p0)
a_1, b_1 = calc_coeffs(start_ned_1, stop_ned_1)

# define line y = a_2*x + b_2 below which points are ignored
stop_ned_2 = pm.geodetic2ned(*(63.43458, 10.40655, 0), *p0)    
start_ned_2 = pm.geodetic2ned(*(63.43312, 10.40514, 0), *p0)    
a_2, b_2 = calc_coeffs(start_ned_2, stop_ned_2)

# # # # # # # # 
start_ned_3 = pm.geodetic2ned(*(63.43184, 10.38101, 0), *p0) 
stop_ned_3 = pm.geodetic2ned(*(63.43282, 10.38580, 0), *p0) 
a_3, b_3 = calc_coeffs(start_ned_3, stop_ned_3)

start_ned_4 = stop_ned_3
stop_ned_4 = pm.geodetic2ned(*(63.43483, 10.39168, 0), *p0) 
a_4, b_4 = calc_coeffs(start_ned_4, stop_ned_4)

start_ned_5 = pm.geodetic2ned(*(63.43533, 10.39318, 0), *p0)
stop_ned_5 = pm.geodetic2ned(*(63.43597, 10.39906, 0), *p0) 
a_5, b_5 = calc_coeffs(start_ned_5, stop_ned_5)

start_ned_6 = stop_ned_5
stop_ned_6 = pm.geodetic2ned(*(63.43579, 10.40397, 0), *p0) 
a_6, b_6 = calc_coeffs(start_ned_6, stop_ned_6)

start_ned_7 = stop_ned_6
stop_ned_7 = pm.geodetic2ned(*(63.43533, 10.40658, 0), *p0)
a_7, b_7 = calc_coeffs(start_ned_7, stop_ned_7)

# ----------
start_ned_8 = pm.geodetic2ned(*(63.43115, 10.37959, 0), *p0)  
stop_ned_8 = pm.geodetic2ned(*(63.43209, 10.38744, 0), *p0) 
a_8, b_8 = calc_coeffs(start_ned_8, stop_ned_8)

start_ned_9 = stop_ned_8
stop_ned_9 = pm.geodetic2ned(*(63.43374, 10.39238, 0), *p0) 
a_9, b_9 = calc_coeffs(start_ned_9, stop_ned_9)

start_ned_10 = stop_ned_9
stop_ned_10 = pm.geodetic2ned(*(63.43385, 10.39343, 0), *p0) 
a_10, b_10 = calc_coeffs(start_ned_10, stop_ned_10)

start_ned_11 = stop_ned_10
stop_ned_11 = pm.geodetic2ned(*(63.43417, 10.39360, 0), *p0) 
a_11, b_11 = calc_coeffs(start_ned_11, stop_ned_11)

start_ned_12 = stop_ned_11
stop_ned_12 = pm.geodetic2ned(*(63.43485, 10.39747, 0), *p0) 
a_12, b_12 = calc_coeffs(start_ned_12, stop_ned_12)

start_ned_13 = stop_ned_12
stop_ned_13 = pm.geodetic2ned(*(63.43503, 10.40283, 0), *p0) 
a_13, b_13 = calc_coeffs(start_ned_13, stop_ned_13)

start_ned_14 = stop_ned_13
stop_ned_14 = pm.geodetic2ned(*(63.43453, 10.40623, 0), *p0)  # [570164, 7034775]
a_14, b_14 = calc_coeffs(start_ned_14, stop_ned_14)

fun1 = [(start_ned_3[1], stop_ned_3[1], a_3, b_3), (start_ned_4[1], stop_ned_4[1], a_4, b_4), (start_ned_5[1], stop_ned_5[1], a_5, b_5), (start_ned_6[1], stop_ned_6[1], a_6, b_6), (start_ned_7[1], stop_ned_7[1], a_7, b_7)]
fun2 = [(start_ned_8[1], stop_ned_8[1], a_8, b_8), (start_ned_9[1], stop_ned_9[1], a_9, b_9), (start_ned_10[1], stop_ned_10[1], a_10, b_10), (start_ned_11[1], stop_ned_11[1], a_11, b_11), (start_ned_12[1], stop_ned_12[1], a_12, b_12), (start_ned_13[1], stop_ned_13[1], a_13, b_13), (start_ned_14[1], stop_ned_14[1], a_14, b_14)]
########

def calc_visited_squares(feature, frequency_table, frequency_table_nb, frequency_table_sb, frequency_table_eb, frequency_table_wb, f_table_pos, f_table_pos_sb, f_table_pos_nb, f_table_pos_eb, f_table_pos_wb, continuation, sid):
    # Calculates the squares in the grid overlay traversed by a ship whose positions are described by a LineString
    # Ship counts are stored in frequency tables
    # Heading and speeds in each grid cell are also stored

    linestring = feature['geometry']['coordinates']
    speed = feature['properties']['speed']
    init_pos_ned = pm.geodetic2ned(*(linestring[0][1], linestring[0][0], 0), *p0)
    end_pos_ned = pm.geodetic2ned(*(linestring[-1][1], linestring[-1][0],0), *p0)
    if end_pos_ned[1] < 796 and end_pos_ned[1] > 721 and end_pos_ned[0] > 447:
        northbound = True
        southbound = False
    elif end_pos_ned[1] < 796 and end_pos_ned[1] > 721 and init_pos_ned[1] < 800 and init_pos_ned[1] > 720 and end_pos_ned[0] < 447:
        southbound = True
        northbound = False
    else:
        northbound = False
        southbound = False
    if end_pos_ned[1] < init_pos_ned[1]:
        westbound = True
    else:
        westbound = False

    for i in range(len(linestring)-1):
        start = linestring[i]
        stop = linestring[i+1]
        if start == stop:
            continue
        start_ned = pm.geodetic2ned(*(start[1], start[0], 0), *p0)
        stop_ned = pm.geodetic2ned(*(stop[1], stop[0], 0), *p0)

        # perform checks
        if start_ned[0] > a_1*start_ned[1] + b_1 or stop_ned[0] > a_1*stop_ned[1] + b_1:
            continue
        if start_ned[0] < a_2*start_ned[1] + b_2 or stop_ned[0] < a_2*stop_ned[1] + b_2:
            continue 

        # y = ax + b
        a = (stop_ned[0] - start_ned[0]) / (stop_ned[1] - start_ned[1])
        b = start_ned[0] - a*start_ned[1]

        if start_ned[1] < stop_ned[1]:
            reverse = False
            low_x = np.ceil(start_ned[1])
            high_x = np.floor(stop_ned[1])
        else:
            reverse = True
            low_x = np.ceil(stop_ned[1])
            high_x = np.floor(start_ned[1])
        if start_ned[0] < stop_ned[0]:
            low_y = np.ceil(start_ned[0])
            high_y = np.floor(stop_ned[0])
        else:
            low_y = np.ceil(stop_ned[0])
            high_y = np.floor(start_ned[0])
        xs = []
        for j in np.arange(low_x, high_x + step, step):
            xs.append((j, 'x'))
        ys = np.arange(low_y, high_y + step, step)

        x_sol = []
        for y in ys:
            x = (y - b) / a
            x_sol.append((x, 'y'))
        xs.extend(x_sol)

        x_ind = np.digitize(start_ned[1], x_bin) - 1
        y_ind = np.digitize(start_ned[0], y_bin) - 1

        heading = np.arctan2(stop_ned[1]-start_ned[1], stop_ned[0]-start_ned[0])
        if heading >= 0:
            heading_ind = np.digitize(heading, heading_bin1)
            heading_c = headings[heading_ind - 1]
        else:
            heading_ind = np.digitize(heading, heading_bin2)
            heading_c = headings[8 + (heading_ind - 1)]

        heading = heading_c

        if isinstance(speed, list):
            if speed[i] > 10:
                continue
            speed_ind = np.digitize(speed[i], speed_bin)
        else:
            if speed > 10:
                continue
            speed_ind = np.digitize(speed, speed_bin)
        speed = speed_bin[speed_ind - 1] + 1

        if i == 0 and continuation == False:
          
            f_table_pos[y_ind, x_ind] += 1
            if northbound:
                f_table_pos_nb[y_ind, x_ind] += 1
                if frequency_table_nb[y_ind, x_ind] == None:
                    frequency_table_nb[y_ind, x_ind] = [(heading, speed)]
                elif frequency_table_nb[y_ind, x_ind] == -1:
                    continue
                else:
                    frequency_table_nb[y_ind, x_ind].append((heading, speed))
            elif southbound:
                f_table_pos_sb[y_ind, x_ind] += 1
                if frequency_table_sb[y_ind, x_ind] == None:
                    frequency_table_sb[y_ind, x_ind] = [(heading, speed)]
                elif frequency_table_sb[y_ind, x_ind] == -1:
                    continue
                else:
                    frequency_table_sb[y_ind, x_ind].append((heading, speed))
            elif westbound:
                f_table_pos_wb[y_ind, x_ind] += 1
                if frequency_table_wb[y_ind, x_ind] == None:
                    frequency_table_wb[y_ind, x_ind] = [(heading, speed)]
                elif frequency_table_wb[y_ind, x_ind] == -1:
                    continue
                else:
                    frequency_table_wb[y_ind, x_ind].append((heading, speed))
            elif westbound == False:
                f_table_pos_eb[y_ind, x_ind] += 1
                if frequency_table_eb[y_ind, x_ind] == None:
                    frequency_table_eb[y_ind, x_ind] = [(heading, speed)]
                elif frequency_table_eb[y_ind, x_ind] == -1:
                    continue
                else:
                    frequency_table_eb[y_ind, x_ind].append((heading, speed))

        if reverse == False:
            xs.sort()
        else:
            xs.sort(reverse=True)

        prev_x = ('nan', 'n')
        for x in xs:
            if x[0] == prev_x[0]:
                if northbound:
                    del frequency_table_nb[y_ind, x_ind][-1]
                elif southbound:
                    del frequency_table_sb[y_ind, x_ind][-1]
                elif westbound:
                    del frequency_table_wb[y_ind, x_ind][-1]
                elif westbound == False:
                    del frequency_table_eb[y_ind, x_ind][-1]

                if prev_x[1] == 'x':
                    if reverse == True and a > 0:
                        y_ind -= 1
                    elif reverse == True and a < 0:
                        y_ind += 1
                    elif reverse == False and a > 0:
                        y_ind += 1
                    else:
                        y_ind -= 1
                else:
                    if reverse == True:
                        x_ind -= 1
                    else:
                        x_ind += 1
            elif x[1] == 'x':
                if reverse == False:
                    x_ind += 1
                else:
                    x_ind -= 1
            else:
                if reverse == False and a > 0:
                    y_ind += 1
                elif reverse == False and a < 0:
                    y_ind -= 1
                elif reverse == True and a > 0:
                    y_ind -= 1
                else: 
                    y_ind += 1

            if northbound:
                f_table_pos_nb[y_ind, x_ind] += 1
                if frequency_table_nb[y_ind, x_ind] == None:
                    frequency_table_nb[y_ind, x_ind] = [(heading, speed)]
                elif frequency_table_nb[y_ind, x_ind] == -1:
                    continue
                else:
                    frequency_table_nb[y_ind, x_ind].append((heading, speed))
            elif southbound:
                f_table_pos_sb[y_ind, x_ind] += 1
                if frequency_table_sb[y_ind, x_ind] == None:
                    frequency_table_sb[y_ind, x_ind] = [(heading, speed)]
                elif frequency_table_sb[y_ind, x_ind] == -1:
                    continue
                else:
                    frequency_table_sb[y_ind, x_ind].append((heading, speed))
            elif westbound:
                f_table_pos_wb[y_ind, x_ind] += 1
                if frequency_table_wb[y_ind, x_ind] == None:
                    frequency_table_wb[y_ind, x_ind] = [(heading, speed)]
                elif frequency_table_wb[y_ind, x_ind] == -1:
                    continue
                else:
                    frequency_table_wb[y_ind, x_ind].append((heading, speed))
            elif westbound == False:
                f_table_pos_eb[y_ind, x_ind] += 1
                if frequency_table_eb[y_ind, x_ind] == None:
                    frequency_table_eb[y_ind, x_ind] = [(heading, speed)]
                elif frequency_table_eb[y_ind, x_ind] == -1:
                    continue
                else:
                    frequency_table_eb[y_ind, x_ind].append((heading, speed))

            prev_x = x
            
    return frequency_table, f_table_pos, f_table_pos_nb, f_table_pos_sb, f_table_pos_eb, f_table_pos_wb, frequency_table_nb, frequency_table_sb, frequency_table_wb, frequency_table_eb

continuation = False
i = 0
sid = 0
for feature in data['features']:
    coords = feature['geometry']['coordinates']
    st = feature['properties']['start']
    if i == 0:
        calc_visited_squares(feature, frequency_table, frequency_table_nb, frequency_table_sb, frequency_table_eb, frequency_table_wb, f_table_pos,f_table_pos_sb, f_table_pos_nb, f_table_pos_eb, f_table_pos_wb, continuation, sid)
        prev_end_coord = coords[-1]
        et = feature['properties']['end']
        i += 1
        continue

    if prev_end_coord == coords[0] and st == et:
        continuation = True
    else:
        continuation = False
        sid += 1
    calc_visited_squares(feature, frequency_table, frequency_table_nb, frequency_table_sb, frequency_table_eb, frequency_table_wb, f_table_pos, f_table_pos_sb, f_table_pos_nb, f_table_pos_eb, f_table_pos_wb, continuation, sid)
    prev_end_coord = coords[-1]
    et = feature['properties']['end']

test_table = np.ones((int(700/step), int(1500/step)))

for i in range(0, frequency_table_nb.shape[1]):
    # i -> x-coord
    for j in range(0, frequency_table_nb.shape[0]):
    # j -> y-coord
        for fun in fun1:
            if y_bin[j] > fun[2]*x_bin[i] + fun[3] and fun[0] <= x_bin[i] <= fun[1]:
                frequency_table_nb[j, i] = int(-1)
                test_table[j,i]=-1
                continue
        for fun in fun2:
            if y_bin[j] < fun[2]*x_bin[i] + fun[3] and fun[0] <= x_bin[i] <= fun[1]:
                frequency_table_nb[j, i] = int(-1)
                test_table[j,i]=-1

for i in range(0, frequency_table_eb.shape[1]):
    # i -> x-coord
    for j in range(0, frequency_table_eb.shape[0]):
    # j -> y-coord
        for fun in fun1:
            if j > fun[2]*i + fun[3] and fun[0] <= i <= fun[1]:
                frequency_table_wb[j,  i] = int(-1)
                continue
        for fun in fun2:
            if j < fun[2]*i + fun[3] and fun[0] <= i <= fun[1]:
                frequency_table_wb[j, i] = int(-1)

                
for i in range(0, frequency_table_eb.shape[1]):
    # i -> x-coord
    for j in range(0, frequency_table_eb.shape[0]):
    # j -> y-coord
        for fun in fun1:
            if y_bin[j] > fun[2]*x_bin[i] + fun[3] and fun[0] <= x_bin[i] <= fun[1]:
                frequency_table_eb[j, i] = int(-1)
                
                continue
        for fun in fun2:
            if y_bin[j] < fun[2]*x_bin[i] + fun[3] and fun[0] <= x_bin[i] <= fun[1]:
                frequency_table_eb[j, i] = int(-1)
                
for i in range(0, frequency_table_sb.shape[1]):
    # i -> x-coord
    for j in range(0, frequency_table_sb.shape[0]):
    # j -> y-coord
        for fun in fun1:
            if y_bin[j] > fun[2]*x_bin[i] + fun[3] and fun[0] <= x_bin[i] <= fun[1]:
                frequency_table_sb[j, i] = int(-1)
                
                continue
        for fun in fun2:
            if y_bin[j] < fun[2]*x_bin[i] + fun[3] and fun[0] <= x_bin[i] <= fun[1]:
                frequency_table_sb[j, i] = int(-1)

frequency_table_tuples_nb = np.empty((int(700/step), int(1500/step)), dtype=object)
frequency_table_tuples_wb = np.empty((int(700/step), int(1500/step)), dtype=object)
frequency_table_tuples_eb = np.empty((int(700/step), int(1500/step)), dtype=object)
frequency_table_tuples_sb = np.empty((int(700/step), int(1500/step)), dtype=object)

for i in range(0, frequency_table_nb.shape[1]):
    # i -> x-coord
    for j in range(0, frequency_table_nb.shape[0]):
    # j -> y-coord

        t_list = frequency_table_nb[j, i]
        
        if t_list == None:
            continue

        if t_list == -1:
            frequency_table_tuples_nb[j, i] = -1
            continue

        if frequency_table_tuples_nb[j, i] == None:
            frequency_table_tuples_nb[j, i] = []
       
        temp = []
        for t in t_list:
            if t in temp:
                continue
            freq = t_list.count(t)
            
            frequency_table_tuples_nb[j,i].append((t[0], t[1], freq))
            temp.append(t)
            
for i in range(0, frequency_table_wb.shape[1]):
    # i -> x-coord
    for j in range(0, frequency_table_wb.shape[0]):
    # j -> y-coord

        t_list = frequency_table_wb[j, i]
        
        if t_list == None:
            continue

        if t_list == -1:
            frequency_table_tuples_wb[j, i] = -1
            continue

        if frequency_table_tuples_wb[j, i] == None:
            frequency_table_tuples_wb[j, i] = []
       
        temp = []
        for t in t_list:
            if t in temp:
                continue
            freq = t_list.count(t)
            
            frequency_table_tuples_wb[j,i].append((t[0], t[1], freq))
            temp.append(t)

for i in range(0, frequency_table_eb.shape[1]):
    # i -> x-coord
    for j in range(0, frequency_table_eb.shape[0]):
    # j -> y-coord

        t_list = frequency_table_eb[j, i]
        
        if t_list == None:
            continue

        if t_list == -1:
            frequency_table_tuples_eb[j, i] = -1
            continue

        if frequency_table_tuples_eb[j, i] == None:
            frequency_table_tuples_eb[j, i] = []
       
        temp = []
        for t in t_list:
            if t in temp:
                continue
            freq = t_list.count(t)
            
            frequency_table_tuples_eb[j,i].append((t[0], t[1], freq))
            temp.append(t)

for i in range(0, frequency_table_sb.shape[1]):
    # i -> x-coord
    for j in range(0, frequency_table_sb.shape[0]):
    # j -> y-coord

        t_list = frequency_table_sb[j, i]
        
        if t_list == None:
            continue

        if t_list == -1:
            frequency_table_tuples_sb[j, i] = -1
            continue

        if frequency_table_tuples_sb[j, i] == None:
            frequency_table_tuples_sb[j, i] = []
       
        temp = []
        for t in t_list:
            if t in temp:
                continue
            freq = t_list.count(t)
            
            frequency_table_tuples_sb[j,i].append((t[0], t[1], freq))
            temp.append(t)  

np.save("frequency_table_tuples_2016-2020_chained_nb_less_s.npy", frequency_table_tuples_nb)
np.save("frequency_table_tuples_2016-2020_chained_eb_less_s.npy", frequency_table_tuples_eb)
np.save("frequency_table_tuples_2016-2020_chained_wb_less_s.npy", frequency_table_tuples_wb)
np.save("frequency_table_tuples_2016-2020_chained_sb_less_s.npy", frequency_table_tuples_sb)

np.savetxt("f_table_pos_2016-2020_chained_eb_less_s.csv", f_table_pos_eb, fmt='%1i',delimiter=',')
np.savetxt("f_table_pos_2016-2020_chained_wb_less_s.csv", f_table_pos_wb, fmt='%1i',delimiter=',')
np.savetxt("f_table_pos_2016-2020_chained_nb_less_s.csv", f_table_pos_nb, fmt='%1i',delimiter=',')
np.savetxt("f_table_pos_2016-2020_chained_sb_less_s.csv", f_table_pos_sb, fmt='%1i',delimiter=',')

np.savetxt("test_table_2016-2020_chained_less_s.csv", test_table, fmt='%1i',delimiter=',')