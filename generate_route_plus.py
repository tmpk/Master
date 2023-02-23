import geojson
import numpy as np
import json
import dateutil.parser
import datetime
import pandas as pd 
import utm
import imagesc
import matplotlib.pyplot as plt
import pymap3d as pm 
import time
from operator import itemgetter
import copy
import imageio
import matplotlib

def display_results():
    res = np.zeros((70,150))
    for waypoint in waypoints[0::1]:
        x = waypoint[0]
        y = waypoint[1]
        i = np.digitize(x, x_bin) - 1
        j = np.digitize(y, y_bin) - 1
        res[j, i] += 1
    #res = np.add(res*10, test_table)
    #res = np.add(res*40, guide[0:140, 0:300]*10)
    res[j_start, i_start] = 5
    res[j_stop, i_stop] = 5
    res = np.add(res, tt)
    fig = imagesc.plot(res, linewidth=0, origin='lower', cmap='viridis')

def wpts_to_geojson(wpts):
    geojson = {'type':'FeatureCollection', 'features':[]}
    feature = {'type':'Feature',
            'properties':{'speed':[],
                          'heading':[]},
            'geometry':{'type':[],
                        'coordinates':[]}}
    feature['geometry']['type'] = 'LineString'
    for wpt in wpts:
        coords = pm.ned2geodetic(*(wpt[1], wpt[0], 0), *p0)
        feature['geometry']['coordinates'].append((coords[1], coords[0]))
        if len(wpt) > 3:
            feature['properties']['speed'].append(wpt[3])
            feature['properties']['heading'].append(wpt[2]*180/np.pi)
        else:
            continue
    geojson['features'].append(feature)
    return geojson

def Step(x, y, i, j, heading, speed):
    #print("step() called")
    while True:
        x_next = x + np.sin(heading)*speed*0.514
        y_next = y + np.cos(heading)*speed*0.514
        i_next = np.digitize(x_next, x_bin) - 1
        j_next = np.digitize(y_next, y_bin) - 1
        if i_next != i  or j_next != j:
            #print("step() exits")
            return x_next, y_next, i_next, j_next
        else:
            x = x_next
            y = y_next
        if x > 1500 or x < 0 or y > 700 or y < 0:
            return x_next, y_next, i_next, j_next

def find_deltas(R):
    deltas = []
    for i in range(R+1):
        j = np.sqrt(R**2 - i**2)
        j = np.int(np.ceil(j))
        if i > 0:
            diff = deltas[-1][1] - j
            if diff > 1:
                j = np.arange(deltas[-1][1]-1, j-1,-1)
                deltas.extend((i,x) for x in j)
            else:
                deltas.append((i,j))
        else:
            deltas.append((i,j))
    new = deltas.copy()
    deltas.extend((t[0],-1*t[1]) for t in new)
    deltas.extend((-1*t[0],-1*t[1]) for t in new)
    deltas.extend((-1*t[0],t[1]) for t in new)
    return deltas

def get_allowed_actions(action_list):
    current_speed = waypoints[-1][3]
    speed_diffs = [np.abs(current_speed - action[1]) for action in action_list]
    min_diff = np.min(speed_diffs)
    speed_inds = np.where(np.logical_and(speed_diffs >= min_diff, speed_diffs <= (min_diff + 2)))[0]
    current_heading = waypoints[-1][2]
    heading_inds = [headings.index(action[0]) for action in action_list]
    heading_diffs = [np.abs(current_heading - headings_rad[heading_ind]) for heading_ind in heading_inds]    
    heading_inds = np.where(np.less(heading_diffs, 0.5))[0] # heading differences less than 45 deg
    action_inds = np.intersect1d(speed_inds, heading_inds)
    return action_inds

def init_LOS_guide(Rs,Q):
    #print("init los called")
    Rs = 5
    initialization = False
    d = 5.0 *1.25   # lookahead distance
    done = False
    x = waypoints[-1][0]
    y = waypoints[-1][1]
    i = np.digitize(x, x_bin) - 1
    j = np.digitize(y, y_bin) - 1
    if i == i_start and j == j_start:
        initialization = True 
        desired_speed_local = desired_speed 
    while True:
        #print("Rs: ",Rs)
        deltas = find_deltas(Rs)
        cs = []
        #res = np.zeros((70,150))
        for delta in deltas:
            v = j + delta[1]
            h = i + delta[0]
            #
            if v > 69:
                v = 69
            if h > 149:
                h = 149
            if v < 0:
                v = 0
            if h < 0:
                h = 0
            pt = guide[v, h]
            #res[v, h] = 1
            if pt == 1:
                cs.append((h,v))
            else:
                continue
        #img = np.add(res,guide)
        #imagesc.plot(img)
        if len(cs) < 2:
            Rs = int(np.rint(1.5 * Rs))
            d = 1.2 * d
        else:
            y_cs = list((c[1] for c in cs))
            x_cs = list((c[0] for c in cs))
            max_y_ind = np.argmax(y_cs)
            min_y_ind = np.argmin(y_cs)
            max_x_ind = np.argmax(x_cs)
            min_x_ind = np.argmin(x_cs)
            if guide_name == 'north_west':
                ref = cs[max_y_ind]
                target = cs[min_y_ind]
            elif guide_name == 'north_east':
                ref = cs[min_x_ind]
                target = cs[max_x_ind]
            elif guide_name == 'north_south':
                ref = cs[max_y_ind]
                target = cs[min_y_ind]
            elif guide_name == 'west_east':
                ref = cs[min_x_ind]
                target = cs[max_x_ind]
            elif guide_name == 'west_north':
                ref = cs[min_y_ind]
                target = cs[max_y_ind]
            elif guide_name == 'east_west':
                ref = cs[max_x_ind]
                target = cs[min_x_ind]
            elif guide_name == 'east_north':
                ref = cs[max_x_ind]
                target = cs[min_x_ind]
            elif guide_name == 'south_north':
                ref = cs[max_y_ind]
                target = cs[min_y_ind]
            else:
                raise Exception('Guide name is ', guide_name)
            if np.sqrt((target[0] - ref[0])**2 + (target[1] - ref[1])**2) <= 5:
                Rs = int(np.rint(1.5 * Rs))
                d = 1.2 * d
            else:
                break
    #print("ref = ", ref)
    #print("target = ", target)
    # --------------
    alpha = np.arctan2(target[0]-ref[0], target[1]-ref[1])    # path tangential angle
    current_heading = waypoints[-1][2]
    wp = []
    while True:
        cte = -(j - target[1])*np.sin(alpha) + (i - target[0])*np.cos(alpha) # cross track error
        ate = (i - target[0])*np.cos(alpha) + (j - target[1])*np.sin(alpha)  # along track error
        chi = alpha + np.arctan(-cte/d) # desired new angle
        #print("ate: ", ate)
        #print("heading: ", chi * 180/np.pi)
        chi_wrapped = np.arctan2(np.sin(chi), np.cos(chi)) # wrap angle to [-180, 180)
        #print("wrapped heading: ", chi_wrapped * 180/np.pi)
        #print("current heading: ", current_heading * 180/np.pi)
        # limit change in angles
        if np.abs(current_heading - chi_wrapped) > 0.524:
            angle = current_heading - np.sign(current_heading - chi_wrapped)*0.524
            heading = np.arctan2(np.sin(angle), np.cos(angle))
        else:
            heading = chi_wrapped
        #print("new heading: ", heading * 180/np.pi)

        if heading >= 0:
            heading_ind = np.digitize(heading, heading_bin1)
            heading_c = headings[heading_ind - 1]
        else:
            heading_ind = np.digitize(heading, heading_bin2)
            heading_c = headings[16 + (heading_ind - 1)]

        heading_ind = headings.index(heading_c)
        heading = headings_rad[heading_ind]
        #print("digitised heading: ", heading * 180/np.pi)
        ###
        current_heading = heading
        #heading_ind = headings_wb.index(heading_c)
        #print("here")
        #heading_ind = headings_eb.index(heading_c) if direction == 'eastbound' else headings_wb.index(heading_c)
        #print("also here")
        #print("digitised heading: ", heading * 180/np.pi)

        #print('cte', cte)
        #print('heading',heading*180/np.pi)
        #print('alpha', alpha*180/np.pi)
        #print('heading', heading*180/np.pi)
        # x_next = x + np.sin(heading)*0.514*desired_speed
        # y_next = y + np.cos(heading)*0.514*desired_speed
        if initialization == True:
            #print("init true line 254")
            speed = desired_speed_local - desired_speed_local * np.abs(ate) / (np.sqrt(ate**2 + 500))
            #print("data: ", data[j,i])
            #print("i,j: ", i,j)
            avg_speed = sum(map(itemgetter(2), data[j,i])) / len(data[j, i]) if data[j, i] != None else 0
            if speed > avg_speed and avg_speed > 0:
                speed = avg_speed

        else: 
            speed = waypoints[-1][3]
        #print("step called")
        #print("speed: ", speed)
        x_next, y_next, i_next, j_next = Step(x, y, i, j, heading, speed)
        speed_ind = np.digitize(speed, speed_bin) - 1
        reward = rewards[j_next, i_next, heading_ind, speed_ind]
        Q[j, i, heading_ind, speed_ind] += alpha_Q * (reward + gamma * np.amax(Q[j_next, i_next, :, :]) - Q[j, i, heading_ind, speed_ind] ) # update Q-table
        #Q[j, i, heading_ind] = 2
        x = x_next  
        y = y_next
        i = i_next
        j = j_next

        #print(x,y)    
        wp.append((x, y, heading, desired_speed))
        current_heading = wp[-1][2]
        # if data[j,i] != None and len(get_allowed_actions(data[j,i])) > 3:
        #     print("init_LOS_guide exiting, got data")
        #     return wp, done
        
        if x > 1500 or y > 700 or x < 0 or y < 0:
            #print("out of bounds")
            raise Exception('Out of bounds')

        if np.sqrt((stop_ned[0] - y)**2 + (stop_ned[1] -x)**2) < 10:
            done = True
            #print('init_LOS_guide exiting, in goal region')
            return wp, done
        
        if ate < 4:
            #print("init_LOS_guide exiting, along-track error < 4")
            return wp, done

        if np.sqrt((target[0]-i)**2 + (target[1]-j)**2) < 4:
            #print("init_LOS_guide exiting")
            return wp, done
    
def end_LOS(ref, target, waypoints, Q):
    #print("end_LOS called")
    d = 5
    x = ref[0]
    y = ref[1]
    i = np.digitize(x, x_bin) - 1
    j = np.digitize(y, y_bin) - 1
    speed = waypoints[-1][3]
    current_heading = waypoints[-1][2]
    #timestamp = waypoints[-1][4]
    alpha = np.arctan2(target[0]-ref[0], target[1]-ref[1])    # path tangential angle
    wp = []
    while True:
        cte = -(y - ref[1])*np.sin(alpha) + (x - ref[0])*np.cos(alpha) # cross track error
        ate = (x - ref[0])*np.cos(alpha) + (y - ref[1])*np.sin(alpha)  # along track error
        chi = alpha + np.arctan(-cte/d)
        #print("heading: ", chi * 180/np.pi)
        chi_wrapped = np.arctan2(np.sin(chi), np.cos(chi))
        #print("wrapped heading: ", chi_wrapped * 180/np.pi)
        #print("current heading: ", current_heading * 180/np.pi)
        if np.abs(current_heading - chi_wrapped) > 0.524:
            angle = current_heading - np.sign(current_heading-chi_wrapped)*0.524
            heading = np.arctan2(np.sin(angle)  , np.cos(angle))
        else:
            heading = chi_wrapped
        #print("new heading: ", heading * 180/np.pi)
        if heading >= 0:
            heading_ind = np.digitize(heading, heading_bin1)
            heading_c = headings[heading_ind - 1]
        else:
            heading_ind = np.digitize(heading, heading_bin2)
            heading_c = headings[16 + (heading_ind - 1)]

        heading_ind = headings.index(heading_c)
        heading = headings_rad[heading_ind]
        #print("digitised heading: ", heading * 180/np.pi)
        ###
        current_heading = heading
        #heading_ind = headings_wb.index(heading_c)
        #heading_ind = headings_eb.index(heading_c) if direction == 'eastbound' else headings_wb.index(heading_c)
        #print("heading_ind wb: ", heading_ind )
        #print(" i j: ", (i, j))
        # x_next = x + np.sin(heading)*speed*0.514
        # y_next = y + np.cos(heading)*speed*0.514
        x_next, y_next, i_next, j_next = Step(x, y, i, j, heading, speed)
        speed_ind = np.digitize(speed, speed_bin) - 1
        reward = rewards[j_next, i_next, heading_ind, speed_ind]
        if i != i_stop and j != j_stop:
            Q[j, i, heading_ind, speed_ind] += alpha_Q * (reward + gamma * np.amax(Q[j_next, i_next, :, :])- Q[j, i, heading_ind, speed_ind] ) # update Q-table
        #Q[j, i, heading_ind] = 1
        dist = np.sqrt((x - target[0])**2 + (y - target[1])**2)

        if i_next == i_stop and j_next == j_stop: #dist < 2:
            wp.append((x, y, heading, speed))
            #print("exiting end_LOS")
            return wp

        # dist = np.sqrt((i - i_stop)**2 + (j - j_stop)**2)
        # if dist < 1:
        #     waypoints.append((x, y, heading, speed))
        #     #print("exiting end_LOS, dist: ", dist)
        #     return
            
        #return waypoints
        x = x_next  
        y = y_next
        i = i_next
        j = j_next
        
        
        if x > 1500 or y > 700 or x < 0 or y < 0:
            raise Exception('Out of bounds')

        wp.append((x, y, heading, speed))
        
def plot_mat(M):
    ll = pm.geodetic2ned(*(63.4309, 10.3772, 0), *p0)
    lr = pm.geodetic2ned(*(63.4309, 10.4073,0), *p0)
    ur = pm.geodetic2ned(*(63.4372, 10.4073, 0), *p0)
    ul = pm.geodetic2ned(*(63.4372, 10.3772,0 ), *p0)
    xs = [ll[1], lr[1], ur[1], ul[1]]
    ys = [ll[0], lr[0], ur[0], ul[0]]
    BBox2 = ((np.min(xs), np.max(xs),
            np.min(ys), np.max(ys),))

    img = imageio.imread('C:/Users/Toni/Documents/Master/figurer/map_of_area.png')
    figM, axM = plt.subplots()
    axM.imshow(img, zorder=0, extent=BBox2, aspect='auto')
    plotM = axM.imshow(M, cmap='viridis',interpolation='nearest', zorder=1, alpha=1.0, extent=BBox2,origin='lower')#imagesc.plot(rewards_plot, linewidth=0, origin='lower', cmap='viridis')
    figM.colorbar(plotM, ax=axM)
    axM.set_aspect('auto')
    plt.show()
    axM.set_xlim([0, 1500])
    axM.set_ylim([0, 700])

area = np.genfromtxt('C:/Users/Toni/Documents/Master/AIS-data/skipshastighet/dataanalysis/test_table_2016_chained.csv')

data_wb = np.load('frequency_table_tuples_2016-2020_chained_wb_less_speeds.npy', allow_pickle=True)
data_eb = np.load('frequency_table_tuples_2016-2020_chained_eb_less_speeds.npy', allow_pickle=True)
data_nb = np.load('frequency_table_tuples_2016-2020_chained_nb_less_speeds.npy', allow_pickle=True)
data_sb = np.load('frequency_table_tuples_2016-2020_chained_sb_less_speeds.npy', allow_pickle=True)
test_table = np.genfromtxt('test_table_2016-2020_chained.csv', delimiter=',')
img = imageio.imread('C:/Users/Toni/Documents/Master/figurer/map_of_area.png')

#print("data size: " ,data_eb.shape[0], data_eb.shape[1])

#ft = np.genfromtxt('f_table_pos.csv',delimiter=',')

p0 = (63.43095, 10.37723, 0) # origin of local coordinate system
step = 10

sect = np.pi/32
heading_bin1 = [0]
heading_bin1.extend([i*sect for i in np.arange(1,32,2)])
heading_bin1.append(np.pi)
heading_bin2 = [-1*x for x in heading_bin1]
heading_bin2.reverse()
# sect = np.pi/16
# heading_bin1 = [0]
# heading_bin1.extend([i*sect for i in np.arange(1,16,2)])
# heading_bin1.append(np.pi)
# heading_bin2 = [-1*x for x in heading_bin1]
# heading_bin2.reverse()
# headings = ['N','NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N']

x_bin = np.arange(0, 1500, step)
y_bin = np.arange(0, 700, step)
headings = ['N', 'NbE','NNE', 'NEbN', 'NE', 'NEbE', 'ENE', 'EbN', 'E', 'EbS', 'ESE', 'SEbE', 'SE', 'SEbS', 'SSE', 'SbE', 'S', 'SbW', 'SSW', 'SWbS', 'SW', 'SWbW', 'WSW', 'WbS', 'W', 'WbN', 'WNW', 'NWbW', 'NW', 'NWbN', 'NNW','NbW', 'N']
# make array of headings [0, pi/32, ..., pi; -15pi/16, ..., -pi/16 ]
headings_rad_1 = np.arange(0,np.pi+np.pi/16,np.pi/16)
headings_rad_2 = np.arange(-np.pi + np.pi/16,0,np.pi/16)
headings_rad = np.append(headings_rad_1, headings_rad_2)
# headings_rad_1 = np.arange(0,np.pi+np.pi/8,np.pi/8)
# headings_rad_2 = np.arange(-np.pi + np.pi/8,0,np.pi/8)
# headings_rad = np.append(headings_rad_1, headings_rad_2)
# speed_bin = np.array(0)
# speed_bin = np.append(speed_bin, np.arange(0.25,32,0.5))
speed_bin = [0, 2, 4, 6, 8, 10] # reduced speed bin
speeds = [1, 3, 5, 7, 9]

# geodetic2ned outputs (North, East, Down)  
# start-stop coordinates for testing:
## east-north
start_ned1 = pm.geodetic2ned(*(63.43536, 10.40452, 0), *p0)
stop_ned1 = pm.geodetic2ned(*(63.43617, 10.39208, 0), *p0)
## east-west
start_ned2 = pm.geodetic2ned(*(63.43525, 10.40506, 0), *p0) 
stop_ned2 = pm.geodetic2ned(*(63.43191, 10.37966, 0), *p0)
## west-east
start_ned3 = pm.geodetic2ned(*(63.43138, 10.38032, 0), *p0) 
#stop_ned3 = pm.geodetic2ned(*(63.43556, 10.40367, 0), *p0)
stop_ned3 = pm.geodetic2ned(*(63.43556, 10.39773, 0), *p0)
## west-north
start_ned4 = pm.geodetic2ned(*(63.43258, 10.38727, 0), *p0) 
stop_ned4 = pm.geodetic2ned(*(63.43633, 10.39234, 0), *p0)
## north-west
start_ned5 = pm.geodetic2ned(*(63.43599, 10.39191, 0), *p0) 
stop_ned5 = pm.geodetic2ned(*(63.43305, 10.38740, 0), *p0)
## north-east
start_ned6 = start_ned5
stop_ned6 = pm.geodetic2ned(*(63.43555, 10.40345, 0), *p0)
## north-south
start_ned7 = start_ned5
stop_ned7 = pm.geodetic2ned(*(63.43412, 10.39306, 0), *p0)
#TEST:
start_ned_t = pm.geodetic2ned(*(63.43493, 10.40606, 0), *p0)
stop_ned_t = pm.geodetic2ned(*(63.43221, 10.37873, 0), *p0)
##
start_ned_tc2 = pm.geodetic2ned(*(63.43494, 10.40584, 0), *p0)
stop_ned_tc2 = pm.geodetic2ned(*(63.43410, 10.39259, 0), *p0)
##
start_ned_tc3 = pm.geodetic2ned(*(63.43523, 10.40480, 0), *p0)
##
start_ned_tc4 = pm.geodetic2ned(*(63.43610, 10.39196, 0), *p0)
stop_ned_tc4 = pm.geodetic2ned(*(63.43180, 10.38337, 0), *p0)
##
start_ned_tc6 = pm.geodetic2ned(*(63.43631, 10.39229, 0), *p0)
stop_ned_tc6 = pm.geodetic2ned(*(63.43514, 10.40470, 0), *p0)
##
start_ned_tc7 = pm.geodetic2ned(*(63.43491, 10.40611, 0), *p0)
stop_ned_tc7 = pm.geodetic2ned(*(63.43334, 10.38825, 0), *p0)
## 
start_ned_tc8 = pm.geodetic2ned(*(63.43180, 10.37959, 0), *p0)
stop_ned_tc8 = pm.geodetic2ned(*(63.43526, 10.39627, 0), *p0)
##
start_ned_tc9 = pm.geodetic2ned(*(63.43185, 10.37953, 0), *p0)
stop_ned_tc9 = pm.geodetic2ned(*(63.43494, 10.40646, 0), *p0)
##
start_ned_tc10 = pm.geodetic2ned(*(63.43493, 10.40667, 0), *p0)
stop_ned_tc10 = pm.geodetic2ned(*(63.43172, 10.38222, 0), *p0)
##
start_ned_tc11 = pm.geodetic2ned(*(63.43429, 10.39308, 0), *p0)
stop_ned_tc11 = pm.geodetic2ned(*(63.43628, 10.39222, 0), *p0)
##
start_ned_tc12 = pm.geodetic2ned(*(63.43182, 10.37954, 0), *p0)
stop_ned_tc12 = pm.geodetic2ned(*(63.43563, 10.39877, 0), *p0)
#-----------------
#tc1: start_ned7, stop_ned6
#tc2: start_ned_tc2, stop_ned_tc2
#tc3: start_ned_tc3, stop_ned1
#tc4: stop_ned_t, stop_ned3
#tc5: start_ned_tc4, stop_ned_tc4
#tc6: stop_ned7, start_ned7
#tc7: start_ned4, stop_ned4
#tc8: start_ned_tc7, stop_ned_tc7
#tc9: start_ned_tc8, stop_ned_tc8
#tc10: start_ned_tc9, stop_ned_tc9
#tc11: start_ned_tc10, stop_ned_tc10
#tc12: start_ned_tc12, stop_ned_tc12
start_ned = stop_ned1
stop_ned = start_ned_t
print('start', start_ned)
print('stop', stop_ned)   
desired_speed = 5.0 

if start_ned[1] < 720 and stop_ned[1] < 796 and stop_ned[1] > 721 and stop_ned[0] > 447:
    guide_name = 'west_north'
    init_heading = np.pi/2
elif start_ned[1] > 800 and  stop_ned[1] < 796 and stop_ned[1] > 721 and stop_ned[0] > 447:
    guide_name = 'east_north' 
    init_heading = -np.pi/2
#------ for southbound sailings:
elif stop_ned[1] < 796 and stop_ned[1] > 721 and start_ned[1] < 800 and start_ned[1] > 720 and stop_ned[0] < 447 and start_ned[0] > 316:
    guide_name = 'north_south'
    init_heading = np.pi
elif start_ned[1] < 800 and start_ned[1] > 720 and stop_ned[1] < 800 and stop_ned[1] > 720 and stop_ned[0] > 447 and start_ned[0] < 447:
    guide_name = 'south_north'
    init_heading = 0
#------ for eastbound sailings:
elif start_ned[1] < 720 and stop_ned[1] > 800:
    guide_name = 'west_east'
    init_heading = np.pi/2
elif start_ned[0] > 460 and start_ned[1] < 800 and start_ned[1] > 720 and stop_ned[1] > 800:
    guide_name = 'north_east'
    init_heading = np.pi
#------ for westbound sailings:
elif start_ned[1] > 800 and stop_ned[1] < 780:
    guide_name = 'east_west' 
    init_heading = -np.pi/2
elif start_ned[0] > 447 and stop_ned[1] < 720:
    guide_name = 'north_west'
    init_heading = -np.pi
else:
    raise Exception('No guide found')

print("guide name:", guide_name)
filename = 'C:/Users/Toni/Documents/Master/AIS-data/skipshastighet/biggercells/LOS_guides/%s_LOS.csv' % guide_name
guide_file = np.genfromtxt(filename,delimiter=',')
x_guide = np.digitize(guide_file[0,:], x_bin)
x_guide = [x - 1 for x in x_guide]
y_guide = np.digitize(guide_file[1,:], y_bin)
y_guide = [y - 1 for y in y_guide]
guide = np.zeros((int(700/step), int(1500/step)))
guide[y_guide, x_guide] = 1
#test_table = np.genfromtxt('test_table_2016-2020_chained.csv', delimiter=',')
#pt = np.add(test_table, guide)
#imagesc.plot(pt, linewidth=0, origin='lower', cmap='jet')
count = 0
if stop_ned[1] < 796 and stop_ned[1] > 721 and stop_ned[0] > 447:
    direction = 'northbound'
    # for i in range(850,1500):
    #     for j in range(0, 700):
    #         if data_nb[j,i] != None and data_wb[j,i] != None and data_wb[j,i] != -1:
    #             data_nb[j,i].extend(data_wb[j,i])
    data = data_nb
elif stop_ned[1] < 796 and stop_ned[1] > 721 and start_ned[1] < 800 and start_ned[1] > 720 and stop_ned[0] < 447:
    direction = 'southbound'
    data = data_sb
elif stop_ned[1] > start_ned[1]:
    direction = 'eastbound'
    data = data_eb
else:
    direction = 'westbound'
    data = data_wb

print("direction:",direction)

prev_action_list = []
x_start = start_ned[1]    
y_start = start_ned[0]
x_stop = stop_ned[1]
y_stop = stop_ned[0]
i_start = np.digitize(x_start, x_bin) - 1 # np.int(np.floor(x_start))
print("i_start: ", i_start)
j_start = np.digitize(y_start, y_bin) - 1 # np.int(np.floor(y_start))
print("j_start: ", j_start)
i_stop = np.digitize(x_stop, x_bin) - 1
print("i_stop: ", i_stop)
j_stop = np.digitize(y_stop, y_bin) - 1
print("j_stop: ", j_stop)

headings_wb = ['S', 'SbW', 'SSW', 'SWbS', 'SW', 'SWbW', 'WSW', 'WbS', 'W', 'WbN', 'WNW', 'NWbW', 'NW', 'NWbN', 'NNW','NbW','N']
wb_action_list = [(h, 4.0, 1) for h in headings_wb]
headings_eb = ['N', 'NbE','NNE', 'NEbN', 'NE', 'NEbE', 'ENE', 'EbN', 'E', 'EbS', 'ESE', 'SEbE', 'SE', 'SEbS', 'SSE', 'SbE', 'S']
eb_action_list = [(h, s, 1) for h in headings_eb for s in speeds]
h_action_list = [(h,s,1) for h in headings for s in speeds]
alpha_Q = 0.1
gamma = 1.0
epsilon = 0.9
Q = np.zeros((70, 150, 32, 5)) 
#Q[j_stop, i_stop, :, :] = 0
print("Q size: ", Q.shape)
grounding = -10
rewards = np.empty((70, 150, 32, 5)) 
rewards_plot = np.empty((70,150))
rewards.fill(-1.0)
max_radius = np.sqrt((i_stop-i_start)**2 + (j_stop-i_start)**2)
print(max_radius)
goal_reward = 100
temp_mat = np.empty((70, 150, 32, 5)) 
temp_mat.fill(-1.0)
for i in range(0,150):
    for j in range(0,70):
        radius = np.sqrt((i_stop - i)**2 + (j_stop - j)**2)
        aux = np.exp(-0.0000001*radius**2) - 1.0
        if data[j,i] != -1 and data[j,i] != None:
            max_val = max(data[j,i], key=itemgetter(2))[2]
            for d in data[j,i]:
                h = headings.index(d[0])

                s = speeds.index(d[1])
                if d[2] > temp_mat[j,i,h,s]:
                    rewards[j, i, h,s] = -(1/d[2])**2 * np.exp(-d[2]/max_val) #+ 0.25*aux #cand: -(1/d[2])*max_val/d[2] + aux #+ 0.25*aux#og:
                    temp_mat[j,i,h,s] = d[2] 
        elif data[j,i] == -1:
            rewards[j,i,:,:] = grounding
        rewards_plot[j,i] = np.amax(rewards[j,i,:,:])
print("min reward: ", np.min(rewards[rewards > -4]))
print("max reward: ", np.max(rewards[rewards > -4]))
rewards_plot[rewards_plot == grounding] = np.nan
rewards_plot[rewards_plot == -2] = np.nan

fig = imagesc.plot(rewards_plot, linewidth=0, origin='lower', cmap='viridis')

rewards[j_stop, i_stop, :, :] = goal_reward

init_heading = 3*np.pi/4
Rs = 20 # search radius
Rl = 5 # lookahead
# t = time.time()
# for ep in range(0,100):
#     waypoints = [(x_start, y_start, init_heading, 0.0)]
#     if data[j_start, i_start] == None:
#         wp, done = init_LOS_guide(Rs,Q)
#         waypoints.extend(wp)
#     x = waypoints[-1][0]
#     y = waypoints[-1][1]
#     count = 0
#     try: 
#         while True:
#             i = np.digitize(x, x_bin) - 1
#             j = np.digitize(y, y_bin) - 1
#             action_list = data[j, i]
#             if action_list == None:
#                 if prev_action_list != None and count == 0:
#                     action_list = prev_action_list
#                     count = 1
#                 else:
#                     wp, done = init_LOS_guide(Rs,Q)
#                     x = wp[-1][0]
#                     y = wp[-1][1]
#                     waypoints.extend(wp)
#                     count = 0
#                     if done:
#                         #waypoints = 
#                         wp = end_LOS((x, y), (stop_ned[1], stop_ned[0]), waypoints,Q)
#                         waypoints.extend(wp)
#                         break
#                     else:
#                         continue
#             a_sum = 0
#             inds = get_allowed_actions(action_list)
#             #print("action lsit: ", action_list)
#             if inds.size == 0:
#                 wp, done = init_LOS_guide(Rs,Q)
#                 x = wp[-1][0]
#                 y = wp[-1][1]
#                 waypoints.extend(wp)
#                 count = 0
#                 if done:
#                     #waypoints = 
#                     wp = end_LOS((x, y), (stop_ned[1], stop_ned[0]), waypoints,Q)
#                     waypoints.extend(wp)
#                     break
#                 else:
#                     continue
#             else:
#                 action_list = [action_list[i] for i in inds]
#             a_sum = np.sum([a[2] for a in action_list])
#             probabilities = [a[2]/a_sum for a in action_list]
#             action_indexes = np.arange(0,len(action_list))
#             #rand() returns a number from a uniform distribution over [0,1), so it will be smaller than...
#             if np.random.rand() < 0.9:
#                 #action_index = np.random.choice(action_indexes, p=probabilities)     # ... epsilon with epsilon probability. Hence we choose greedy action with epsilon probability
#                 action_index = np.argmax(probabilities)
#                 action = action_list[action_index]
#                 #print("greedy action: ", action)
#             else:
#                 action_index = np.random.choice(action_indexes, p=probabilities)
#                 action = action_list[action_index]
#                 #print("random action: ", action)
#             #print("action: ",action[0])
#             heading_ind = headings.index(action[0])
#             heading_rad = headings_rad[heading_ind]
#             speed = action[1]
#             speed_ind = speeds.index(speed) 

#             x_next, y_next, i_next, j_next = Step(x, y, i, j, heading_rad, speed)
#             reward = rewards[j_next, i_next, heading_ind, speed_ind]
#             Q[j, i, heading_ind, speed_ind] += alpha_Q * (reward + gamma * np.amax(Q[j_next, i_next, :, :]) - Q[j, i, heading_ind, speed_ind] ) # update Q-table
#             x = x_next
#             y = y_next
#             waypoints.append((x, y, heading_rad, speed))
            
#             #print("heading: ", heading_rad * 180/np.pi)
#             prev_action_list = action_list
#             if i_next >= 300 or j_next >= 140 or i_next < 0 or j_next < 0:
#                 raise Exception('Out of bounds')
            
#             if np.sqrt((stop_ned[0] - y)**2 + (stop_ned[1] -x)**2) < 50:
#                 wp = end_LOS((x, y), (stop_ned[1], stop_ned[0]), waypoints,Q)
#                 waypoints.extend(wp)
#                 break
        
#     except Exception:
#         #print("exc")
#         continue
# elapsed = time.time() - t
# print("time elapsed: ", elapsed)

# #display_results()

# #print(waypoints[-1])        
# res = np.zeros((int(700/step),int(1500/step)))

# for waypoint in waypoints[0::5]:
#     x = waypoint[0]
#     y = waypoint[1]
#     i = np.digitize(x, x_bin) - 1
#     j = np.digitize(y, y_bin) - 1
#     res[j, i] += 1

#res = np.add(res*10, test_table)

ll = pm.geodetic2ned(*(63.4309, 10.3772, 0), *p0)
lr = pm.geodetic2ned(*(63.4309, 10.4073,0), *p0)
ur = pm.geodetic2ned(*(63.4372, 10.4073, 0), *p0)
ul = pm.geodetic2ned(*(63.4372, 10.3772,0 ), *p0)
xs = [ll[1], lr[1], ur[1], ul[1]]
ys = [ll[0], lr[0], ur[0], ul[0]]
BBox = ((10.3772,   10.4073,      
         63.4309, 63.4372))
BBox2 = ((np.min(xs), np.max(xs),
          np.min(ys), np.max(ys),))

V = np.zeros((int(700/step),int(1500/step)))
print("min Q: ", np.amin(Q))
print("max Q: ", np.amax(Q))
print("np.average(Q):", np.average(Q[np.logical_and(Q > -1.0, Q < 0.0)]))
Q[Q == 0] = -1.0
#Q[rewards == grounding] = -10
V = np.amax(Q, axis = (2,3))
V[V == -10] = np.nan
#res = np.add(res*10, V)
#res = np.add(res, ft)
#wpts_geojson = wpts_to_geojson(waypoints[0::5])
# figN, axN = plt.subplots()
# plt.imshow(img, zorder=0, extent=BBox2, aspect='auto')
# axN.imshow(V, cmap='viridis',interpolation='nearest', zorder=1, alpha=0.8, extent=BBox2,origin='lower')#imagesc.plot(rewards_plot, linewidth=0, origin='lower', cmap='viridis')
# plt.show()
plot_mat(V)
Q = np.zeros((140,300,32,5))
waypoints = []
waypoints = [(x_start, y_start, init_heading, desired_speed)]


# geojson_str = geojson.dumps(wpts_geojson, indent=2)
# output_filename = 'waypoints.geojson'
# with open(output_filename, 'w') as output_file:
#     output_file.write('{}'.format(geojson_str))

# guide_geojson = wpts_to_geojson(list(zip(x_guide, y_guide)))
# geojson_str2 = geojson.dumps(guide_geojson, indent=2)
# output_filename = 'guide.geojson'
# with open(output_filename, 'w') as output_file:
#     output_file.write('{}'.format(geojson_str2))

rg = 0
t = time.time()

wb_action_list = [(h, s, 1) for h in headings_wb for s in speeds]

def epsilon_greedy(current_heading):
    #print(epsilon)
    #action_inds = get_allowed_actions2(wb_action_list, current_heading) #allowed inds to wb lsit
    action_inds = get_allowed_actions2(h_action_list, current_heading) 

    if np.random.rand() < epsilon:                   # rand() returns a number from a uniform distribution over [0,1), so it will be smaller than...        
            if np.sum(Q[j,i,:,:]) == 0:
                action_ind = np.random.choice(action_inds) # np.random.choice(range(0,17))
                #action = wb_action_list[action_ind]
                action = h_action_list[action_ind] # if direction == 'eastbound' else wb_action_list[action_ind]
                heading = action[0]
                speed = action[1]
                heading_ind = headings.index(heading)
                #heading_ind = headings_eb.index(heading) if direction == 'eastbound' else headings_wb.index(heading)
                speed_ind = speeds.index(speed)
            else:
                #print("greedy")
                action_index = np.random.choice(np.flatnonzero(Q[j,i,:,:] == Q[j,i,:,:].max()))#np.argmax(Q[j, i, :,:])
                action_indices = np.unravel_index(action_index, Q[j,i,:,:].shape)
                heading_ind = action_indices[0]
                #heading = headings_wb[heading_ind]
                heading = headings[heading_ind] #if direction == 'eastbound' else headings_wb[heading_ind] 
                speed_ind = action_indices[1]
                speed = speeds[speed_ind]
                #print("action_inds: ", action_index)
                #print("Q choices: ", Q[j, i, action_inds]) 
    else:
        action_ind = np.random.choice(action_inds) # np.random.choice(range(0,17))
        action = h_action_list[action_ind]
        #action = eb_action_list[action_ind] if direction == 'eastbound' else wb_action_list[action_ind]
        heading = action[0]
        speed = action[1]
        #heading_ind = headings_wb.index(heading)
        heading_ind = headings.index(heading) # if direction == 'eastbound' else headings_wb.index(heading)
        speed_ind = speeds.index(speed)
    return heading_ind, heading, speed_ind, speed


def get_allowed_actions2(action_list, current_heading):
    current_speed = waypoints[-1][3]
    speed_diffs = [np.abs(current_speed - action[1]) for action in action_list] 
    min_diff = np.min(speed_diffs)
    speed_inds = np.where(np.logical_and(speed_diffs >= min_diff, speed_diffs <= (min_diff + 2)))[0]
    #print(" c h: ", current_heading)
    heading_inds = [headings.index(action[0]) for action in action_list]
    heading_diffs = [np.abs(current_heading - headings_rad[heading_ind]) for heading_ind in heading_inds]    
    heading_inds2 = np.where(np.less(heading_diffs, 0.5))[0] # heading differences less than 30 deg
    action_inds = np.intersect1d(speed_inds, heading_inds2)
    return action_inds  

paths=np.zeros((70,150))
path_x = []
path_y = []
wp_out = []
episode_reward = 0
episode_rewards = []
cumulative_reward = 0
cumulative_rewards = []
episode_episode = []
times_reached_goal = 0

for e in range(0,15000):
    if e % 100 == 0:
        print("ep: ", e)
    x = start_ned[1]
    y = start_ned[0]
    current_heading = init_heading
    prev_heading = np.nan
    while True:
        i = np.digitize(x, x_bin) - 1
        j = np.digitize(y, y_bin) - 1
        #print(i,j)
        heading_ind, heading, speed_ind, speed = epsilon_greedy(current_heading)
        heading_rad = headings_rad[headings.index(heading)]
        #("i,j: ", i,j)
        #print("heading_inds, heading, speed_ind: ", heading_ind, heading, speed_ind)
        x_next, y_next, i_next, j_next = Step(x, y, i, j, heading_rad, speed)

        reward = rewards[j_next, i_next, heading_ind, speed_ind]
        episode_reward += reward
        cumulative_reward += reward
        current_heading = heading_rad

        # if prev_heading != np.nan:
        #     # delta = np.abs(current_heading - prev_heading)
        #     # reward += delta * -0.1
        #     if prev_heading != current_heading:
        #         reward += -0.0001

        Q[j, i, heading_ind, speed_ind] += alpha_Q * (reward + gamma * np.amax(Q[j_next, i_next, :, :])- Q[j, i, heading_ind, speed_ind] ) # update Q-table
        x = x_next
        y = y_next
        waypoints.append((x,y,heading_rad,desired_speed))
        path_x.append(i)
        path_y.append(j)
        paths[j,i] = 1

        if i_next > 298 or j_next > 138 or i<0 or j<0:
            #print("out of bounds, breaking")
            break
        
        if reward == goal_reward:
            rg+=1
            
            if epsilon < 0.9:
                epsilon += 0.05
            #print("Goal reached in ep: ", e)
            #if e % 25 == 0:
                #plt.scatter(path_x, path_y, label='episode = %.2f ' %e)
                #lt.legend()
            times_reached_goal += 1
  
            break
        

        if data[j,i] == -1:
            break

        prev_heading = current_heading

    episode_rewards.append(episode_reward/(e+1))
    episode_episode.append(e)
    cumulative_rewards.append(cumulative_reward)
    if e != 49999:
        waypoints = [(x_start, y_start, init_heading, desired_speed)]
    path_x = []
    path_y = []
    
    paths = np.zeros((70,150))

elapsed = time.time() - t
print("time elapsed: ", elapsed)
print("times reached goal: ", rg)

np.save("Q_table_H_tc1_5k", Q)

res = np.zeros((int(700/step),int(1500/step)))
res_ult = np.zeros((70,150))
res = np.amax(Q, axis = (2,3))
res[res == 0] = np.nan
res[res == -1] = np.nan
plot_mat(res)
#res[j_stop, i_stop] = 10
#######
########

##


plot_CR = plt.plot(episode_episode, cumulative_rewards)
plt.xlabel("Episode")
plt.ylabel("Cumulative reward")
plt.show()
#
plot_AR = plt.plot(episode_episode, episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Average reward")
plt.show()
# extract optimum
opt = np.zeros((70,150))
opt[j_start, i_start] = 1

fig1, ax1 = plt.subplots()
prev_action_indexes = []

opt_wpts = []

i = i_start
x = x_start
y = y_start
j = j_start

while i < i_stop:
    action_index = np.argmax(Q[j,i,:,:])
    action_indices = np.unravel_index(action_index, Q[j,i,:,:].shape)
    heading_ind = action_indices[0]
    speed_ind = action_indices[1]
    heading_rad = headings_rad[heading_ind]
    #heading_rad = headings_rad[headings.index(headings_eb[heading_ind])] if direction == 'eastbound' else headings_rad[headings.index(headings_wb[heading_ind])]
    speed = speeds[speed_ind]
    #print(speed)
    x_next, y_next, i_next, j_next = Step(x, y, i, j, heading_rad, speed)
    opt[j_next, i_next] = 1
    x = x_next
    y = y_next
    i = i_next
    j = j_next
    opt_wpts.append((x, y, heading_rad, speed))
    ax1.scatter(i,j, color='blue')
    print("i,j: ", i,j)

waypoints = opt_wpts
temp = waypoints[::5]
if temp[-1] == waypoints[-1]:
    waypoints = temp
else:
    temp.append(waypoints[-1])
    waypoints = temp
vec_xs = np.zeros(len(waypoints))
vec_ys = np.zeros(len(waypoints))
vec_u = np.zeros(len(waypoints))
vec_v = np.zeros(len(waypoints))
velocities = np.zeros(len(waypoints))
fig, ax = plt.subplots()
plt.imshow(img, zorder=0, extent=BBox2, aspect='auto')
for i in np.arange(0,len(waypoints)):
    vec_xs[i] = waypoints[i][0] 
    vec_ys[i] = waypoints[i][1]
    vec_u[i] = np.sin(waypoints[i][2]) # waypoints[i][3] *
    vec_v[i] = np.cos(waypoints[i][2]) # waypoints[i][3] * 
    velocities[i] = waypoints[i][3]
for i in np.arange(1, len(vec_u)):
    v_u = vec_u[i]
    v_v = vec_v[i]
    vec_u[i] = v_u / np.sqrt(v_u**2 + v_v**2)
    vec_v[i] = v_v / np.sqrt(v_u**2 + v_v**2)

# ax.quiver(vec_xs[0:len(vec_xs):10], vec_ys[0:len(vec_ys):10], vec_u[0:len(vec_u):10], vec_v[0:len(vec_v):10], velocities[0:len(velocities):10],
#             units='width', angles='xy', scale_units='x', scale=0.045, headwidth=2.0, headlength=2.0,
#             headaxislength=1.0, minlength=0.1, width=0.01, cmap='Wistia', pivot='mid')
# norm = matplotlib.colors.Normalize(vmin=np.min(velocities[0:len(velocities):10]), vmax=np.max(velocities[0:len(velocities):10]),clip=False)
# sm = matplotlib.cm.ScalarMappable(cmap='Wistia', norm=norm)
# cbar = fig.colorbar(sm)
# cbar.set_label('knots', rotation=270, labelpad=15)
# ax.set_ylim([0,700])
# ax.set_xlim([0,1500])
# ax.set_title('Learned waypoints with heading and speed indicated.')
# print(velocities[0:len(velocities):10])
# plt.show()

print("lengths: ", np.sqrt(np.add(vec_u**2, vec_v**2)))
v_diff = np.max(velocities)-np.min(velocities)
cmap = matplotlib.cm.get_cmap('Wistia', 1) if v_diff == 0 else matplotlib.cm.get_cmap('Wistia', v_diff / 2 + 1)
norm = matplotlib.colors.Normalize(vmin=np.min(velocities)-1, vmax=np.max(velocities)+1,clip=False)
sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
ax.quiver(vec_xs, vec_ys, vec_u, vec_v, velocities,
            units='width', angles='xy', scale_units='x', scale=0.045, headwidth=2.0, headlength=2.0,
            headaxislength=1.0, minlength=0.1, width=0.01, cmap=cmap, pivot='mid')
cbar = fig.colorbar(sm, ticks=np.arange(np.min(velocities),np.max(velocities)+1,2))
cbar.set_label('knots', rotation=270, labelpad=15)
ax.set_ylim([0,700])
ax.set_xlim([0,1500])
ax.set_title('Learned waypoints with heading and speed indicated.')
print(velocities)
plt.show()

out = []
for waypoint in waypoints:
    x = waypoint[0]    # easting
    y = waypoint[1]    # northing
    heading = waypoint[2]
    speed = waypoint[3]
    lla = pm.ned2geodetic(*(y,x,0), *p0) # lla: (lat, lon, alt)
    out.append((lla[0], lla[1], heading, speed))

np.savetxt("waypoints_tc13", out, fmt='%1.5f',delimiter=',')

wpts_geojson = wpts_to_geojson(waypoints)
geojson_str = geojson.dumps(wpts_geojson, indent=2)
output_filename = 'waypoints_tc13.geojson'
with open(output_filename, 'w') as output_file:
    output_file.write('{}'.format(geojson_str))