import numpy as np
import geojson
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pymap3d as pm
import csv

df1 = pd.read_excel (r'skipshastighet_2016/all/Liste_over_seilas-202005041453.xlsx')
df2 = pd.read_excel (r'skipshastighet_2017/all/Liste over seilas-202005041457.xlsx')
df3 = pd.read_excel (r'skipshastighet_2018/all/Liste over seilas-202005041500.xlsx')
df4 = pd.read_excel (r'skipshastighet_2019/all/Liste over seilas-202006031755.xlsx')
df5 = pd.read_excel (r'skipshastighet_2020_jan-mai/ravnkloa/Liste over seilas-202006041847.xlsx')
avg_speeds1 = np.array(df1['Gjsn hast.'])
avg_speeds2 = np.array(df2['Gjsn hast.'])
avg_speeds3 = np.array(df3['Gjsn hast.'])
avg_speeds4 = np.array(df4['Gjsn hast.'])
avg_speeds5 = np.array(df5['Gjsn hast.'])
avg_speeds = np.concatenate((avg_speeds1, avg_speeds2, avg_speeds3, avg_speeds4, avg_speeds5))
avg_speeds = avg_speeds[avg_speeds < 45]
p0 = (63.43095, 10.37723, 0) # origin local ned system

# # # # # # # # 
def calc_coeffs(start, stop):
    # returns coefficients a, b for the linear interpolated line y = ax + b between "start", "stop";  points are in NED-frame
    a = (stop[0] - start[0]) / (stop[1] -start[1])
    b = start[0] - a*start[1]
    return a, b

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

def df_to_geojson(df, properties):
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

with open('dataset_2016-2020_chained.geojson') as f1:
     data = geojson.load(f1)

# out_w_n = []
# out_w_e = []
# out_e_w = []
# out_e_n = []
# out_s_n = []
# out_n_s = []
# out_n_e = []
# out_n_w = []
# skip = False

# direction = ''

# out_hs = []
# for feature in data['features']:
#      linestring = feature['geometry']['coordinates']
#      init_pos_ned = pm.geodetic2ned(*(linestring[0][1], linestring[0][0], 0), *p0)
#      end_pos_ned = pm.geodetic2ned(*(linestring[-1][1], linestring[-1][0],0), *p0)
#      for fun in fun1:
#           if init_pos_ned[0] > fun[2]*init_pos_ned[1] + fun[3] and fun[0] <= init_pos_ned[1] <= fun[1]:
#                skip = True
#                break
#           elif end_pos_ned[0] > fun[2]*end_pos_ned[1] + fun[3] and fun[0] <= end_pos_ned[1] <= fun[1]:
#                skip = True
#                break

#      if skip:
#           skip = False
#           continue

#      for fun in fun2:
#           if end_pos_ned[0] < fun[2]*end_pos_ned[1] + fun[3] and fun[0] <= end_pos_ned[1] <= fun[1]:
#                skip = True
#                break
#           elif init_pos_ned[0] < fun[2]*init_pos_ned[1] + fun[3] and fun[0] <= init_pos_ned[1] <= fun[1]:
#                skip = True
#                break

#      if skip:
#           skip = False
#           continue

#      speeds = feature['properties']['speed']
#      if isinstance(speeds, float) or isinstance(speeds, int):
#           continue
#      for speed in speeds:
#           if speed > 12:
#                out_hs.append(feature)
#                break

# gj = {'type':'FeatureCollection', 'features':[]}
# gj['features'].extend([elem for elem in out_hs])
# geojson_str = geojson.dumps(gj, indent=2)
# output_filename = 'dataset_2016-2020_chained_high_speeds.geojson'

# with open(output_filename, 'w') as output_file:
#     output_file.write('{}'.format(geojson_str))

# for feature in data['features']:
#      linestring = feature['geometry']['coordinates']
#      init_pos_ned = pm.geodetic2ned(*(linestring[0][1], linestring[0][0], 0), *p0)
#      end_pos_ned = pm.geodetic2ned(*(linestring[-1][1], linestring[-1][0],0), *p0)
#      for fun in fun1:
#           if init_pos_ned[0] > fun[2]*init_pos_ned[1] + fun[3] and fun[0] <= init_pos_ned[1] <= fun[1]:
#                skip = True
#                break
#           elif end_pos_ned[0] > fun[2]*end_pos_ned[1] + fun[3] and fun[0] <= end_pos_ned[1] <= fun[1]:
#                skip = True
#                break

#      if skip:
#           skip = False
#           continue

#      for fun in fun2:
#           if end_pos_ned[0] < fun[2]*end_pos_ned[1] + fun[3] and fun[0] <= end_pos_ned[1] <= fun[1]:
#                skip = True
#                break
#           elif init_pos_ned[0] < fun[2]*init_pos_ned[1] + fun[3] and fun[0] <= init_pos_ned[1] <= fun[1]:
#                skip = True
#                break

#      if skip:
#           skip = False
#           continue
     
#      #------ for northbound sailings:
#      if init_pos_ned[1] < 720 and end_pos_ned[1] < 796 and end_pos_ned[1] > 721 and end_pos_ned[0] > 447:
#           direction = 'west_north'
#      elif init_pos_ned[1] > 800 and  end_pos_ned[1] < 796 and end_pos_ned[1] > 721 and end_pos_ned[0] > 447:
#           direction = 'east_north' 
#      #------ for southbound sailings:
#      elif end_pos_ned[1] < 796 and end_pos_ned[1] > 721 and init_pos_ned[1] < 800 and init_pos_ned[1] > 720 and end_pos_ned[0] < 447 and init_pos_ned[0] > 316:
#           direction = 'north_south'
#      elif init_pos_ned[1] < 800 and init_pos_ned[1] > 720 and end_pos_ned[1] < 800 and end_pos_ned[1] > 720 and end_pos_ned[0] > 447 and init_pos_ned[0] < 447:
#           direction = 'south_north'
#      #------ for eastbound sailings:
#      elif init_pos_ned[1] < 720 and end_pos_ned[1] > 800:
#           direction = 'west_east'
#      elif init_pos_ned[0] > 460 and init_pos_ned[1] < 800 and init_pos_ned[1] > 720 and end_pos_ned[1] > 800:
#           direction = 'north_east'
#      #------ for westbound sailings:
#      elif init_pos_ned[1] > 800 and end_pos_ned[1] < 720:
#           direction = 'east_west'
#      elif init_pos_ned[0] > 447 and end_pos_ned[1] < 720:
#           direction = 'north_west'
#      else:
#           continue
          
#      if direction == 'west_east':
#           out_w_e.append(feature)  
#           for coords in linestring:
#                coords_ned = pm.geodetic2ned(*(coords[1], coords[0], 0), *p0)
#                if coords_ned [1] > 720 and coords_ned[1] < 800 and coords_ned[0] > 447 or coords_ned[1] > 1400 and coords_ned[0] < 388:
#                     del out_w_e[-1]
#                     break
#      elif direction == 'east_west':
#           out_e_w.append(feature)  
#           for coords in linestring:
#                coords_ned = pm.geodetic2ned(*(coords[1], coords[0], 0), *p0)
#                if coords_ned[1] > 1440 and coords_ned[0] > 460:
#                     del out_e_w[-1]
#                     break
#      elif direction == 'west_north':
#           out_w_n.append(feature)  
#           for coords in linestring:
#                coords_ned = pm.geodetic2ned(*(coords[1], coords[0], 0), *p0)
#                if coords_ned[1] > 800:
#                     del out_w_n[-1]
#                     break
#      elif direction == 'east_north':
#           out_e_n.append(feature)  
#           for coords in linestring:
#                coords_ned = pm.geodetic2ned(*(coords[1], coords[0], 0), *p0)
#                if coords_ned[1] < 720 or coords_ned[0] < 370:
#                     del out_e_n[-1]
#                     break
#      elif direction == 'north_west':
#           out_n_w.append(feature)  
#           for coords in linestring:
#                coords_ned = pm.geodetic2ned(*(coords[1], coords[0], 0), *p0)
#                if coords_ned[1] > 800:
#                     del out_n_w[-1]
#                     break
#      elif direction == 'north_east':
#           out_n_e.append(feature)  
#           for coords in linestring:
#                coords_ned = pm.geodetic2ned(*(coords[1], coords[0], 0), *p0)
#                if coords_ned[1] < 720 or (coords_ned[0]<420) or coords_ned[1] > 1450:
#                     del out_n_e[-1]
#                     break
#      elif direction == 'north_south':
#           out_n_s.append(feature)  
#           for coords in linestring:
#                coords_ned = pm.geodetic2ned(*(coords[1], coords[0], 0), *p0)
#                if coords_ned[0] < 316:
#                     del out_n_s[-1]
#                     break
#      elif direction == 'south_north':
#           out_s_n.append(feature)
#           for coords in linestring:
#                coords_ned = pm.geodetic2ned(*(coords[1], coords[0], 0), *p0)
#                if coords_ned[0] < 316:
#                     del out_s_n[-1]
#                     break
#      #-------

# #-------------------------

# east_w_n = []
# north_w_n = []

# east_n_w = []
# north_n_w = []

# east_n_e = []
# north_n_e = []

# east_n_s = []
# north_n_s = []

# east_s_n = []
# north_s_n = []

# east_w_e = []
# north_w_e = []

# east_e_w = []
# north_e_w = []

# east_e_n = []
# north_e_n = []

# for feature in out_n_e:
#      coords = feature['geometry']['coordinates']
#      for c in coords:
#           c = pm.geodetic2ned(*(c[1],c[0],0), *p0)
#           east_n_e.append(c[1])
#           north_n_e.append(c[0])

# for feature in out_n_s:
#      coords = feature['geometry']['coordinates']
#      for c in coords:
#           c = pm.geodetic2ned(*(c[1],c[0],0), *p0)
#           east_n_s.append(c[1])
#           north_n_s.append(c[0])

# for feature in out_n_w:
#      coords = feature['geometry']['coordinates']
#      for c in coords:
#           c = pm.geodetic2ned(*(c[1],c[0],0), *p0)
#           east_n_w.append(c[1])
#           north_n_w.append(c[0])

# for feature in out_w_n:
#      coords = feature['geometry']['coordinates']
#      for c in coords:
#           c = pm.geodetic2ned(*(c[1],c[0],0), *p0)
#           east_w_n.append(c[1])
#           north_w_n.append(c[0])

# for feature in out_e_n:
#      coords = feature['geometry']['coordinates']
#      for c in coords:
#           c = pm.geodetic2ned(*(c[1],c[0],0), *p0)
#           east_e_n.append(c[1])
#           north_e_n.append(c[0])

# for feature in out_e_w:
#      coords = feature['geometry']['coordinates']
#      for c in coords:
#           c = pm.geodetic2ned(*(c[1],c[0],0), *p0)
#           east_e_w.append(c[1])
#           north_e_w.append(c[0])

# for feature in out_w_e:
#      coords = feature['geometry']['coordinates']
#      for c in coords:
#           c = pm.geodetic2ned(*(c[1],c[0],0), *p0)
#           east_w_e.append(c[1])
#           north_w_e.append(c[0])
# for feature in out_s_n:
#      coords = feature['geometry']['coordinates']
#      for c in coords:
#           c = pm.geodetic2ned(*(c[1],c[0],0), *p0)
#           east_s_n.append(c[1])
#           north_s_n.append(c[0])

# plt.scatter(east_s_n, north_s_n, marker=".")
# plt.xlim(0,1500)
# plt.ylim(0,700)
# plt.show()
# north_east = [east_n_e, 
#                north_n_e]
# north_west = [east_n_w, 
#                north_n_w]
# north_south = [east_n_s, 
#                north_n_s]
# west_east = [east_w_e, 
#                north_w_e]
# west_north = [east_w_n, 
#                north_w_n]
# east_north = [east_e_n, 
#                north_e_n]
# east_west = [east_e_w, 
#                north_e_w]
# south_north = [east_s_n, 
#                north_s_n]


# with open("north_east.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(north_east)

# with open("north_west.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(north_west)
    
# with open("north_south.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(north_south)
    
# with open("west_east.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(west_east)
    
# with open("west_north.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(west_north)
    
# with open("east_north.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(east_north)
    
# with open("east_west.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(east_west)
    
# with open("south_north.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(south_north)


# -------calculate histogram and pdf for speeds
avg_speeds_emp = []
all_rec_speeds = []
tmp = 0
speeds_distances = []
for feature in data['features']:
     #---- remove data outside region of interest
     linestring = feature['geometry']['coordinates']
     init_pos_ned = pm.geodetic2ned(*(linestring[0][1], linestring[0][0], 0), *p0)
     end_pos_ned = pm.geodetic2ned(*(linestring[-1][1], linestring[-1][0],0), *p0)
     for fun in fun1:
          if init_pos_ned[0] > fun[2]*init_pos_ned[1] + fun[3] and fun[0] <= init_pos_ned[1] <= fun[1]:
               skip = True
               break
          elif end_pos_ned[0] > fun[2]*end_pos_ned[1] + fun[3] and fun[0] <= end_pos_ned[1] <= fun[1]:
               skip = True
               break

     if skip:
          skip = False
          continue

     for fun in fun2:
          if end_pos_ned[0] < fun[2]*end_pos_ned[1] + fun[3] and fun[0] <= end_pos_ned[1] <= fun[1]:
               skip = True
               break
          elif init_pos_ned[0] < fun[2]*init_pos_ned[1] + fun[3] and fun[0] <= init_pos_ned[1] <= fun[1]:
               skip = True
               break

     if skip:
          skip = False
          continue
     #----- end removal
     
     speeds = feature['properties']['speed']
     speeds_sum = np.sum(speeds)
     coords_ned = [pm.geodetic2ned(*(ls[1], ls[0], 0), *p0) for ls in linestring] 
     distances = []
     if np.max(speeds) > 45:
          continue
     #print("feature: ", feature)
     for ind in np.arange(0,len(coords_ned)-1):
         distances.append(int(np.rint((np.sqrt((coords_ned[ind+1][0] - coords_ned[ind][0])**2 + (coords_ned[ind+1][1] - coords_ned[ind][1])**2))/10)))
     if isinstance(speeds, list):
          speeds_distances = list(zip(speeds, distances))
     else:
          speeds_distances = [(speeds, distances[0])]
     # if isinstance(speeds, list):
     #    for s in speeds:
     #        if s < 45:
     #            all_rec_speeds.append(s)
     # else:
     #    if speeds < 45:
     #        all_rec_speeds.append(speeds)
     #print("line 461: ", speeds_distances)
     

     #all_rec_speeds.extend( [elem[0]] * elem[1] for elem in speeds_distances )
     for elem in speeds_distances:
          all_rec_speeds.extend( [elem[0]] * elem[1] )
     if tmp == 0:
          print(speeds_distances)
          print(all_rec_speeds)
          tmp += 1
     avg_speed = speeds_sum / len(speeds) if isinstance(speeds, list) else speeds_sum 
     if avg_speed > 45:
          continue
     avg_speeds_emp.append(avg_speed)

# coords_ned = [pm.geodetic2ned(*(ls[0][1], ls[0][0], 0), *p0) for ls in linestring] 
#     distances = []
#     for ind in np.arange(0,len(coords_ned-1)):
#          distances.append(np.sqrt((coords_ned[ind+1][0] - coords_ned[ind][0])**2 + (coords_ned[ind+1][1] - coords_ned[ind][1])**2))
#     speeds_distances.append(list(zip(speeds, distances)))

print("min speed", np.min(all_rec_speeds))
print("max speed", np.max(all_rec_speeds))
print("len all", len(all_rec_speeds))
print("min avg speed", np.min(avg_speeds_emp))
print("max avg speed", np.max(avg_speeds_emp))
print("len avg", len(avg_speeds_emp))

bind = np.arange(0, 40, 1)
plt.hist(all_rec_speeds, bins=bind, density=True)
xlabels = ['']*len(bind)
ylabels = ['']*len(np.arange(0.0,0.30,0.01))
c = 0
for i in np.arange(0, 40, 5):
    xlabels[i] = i
for i in np.arange(0.0, 0.30, 0.05):
    ylabels[c] = i
    c+=5
ylabels[15] = 0.15
plt.xticks(bind, xlabels)
plt.yticks(np.arange(0.0, 0.30, 0.01), ylabels)
plt.ylabel('Relative frequency')
plt.xlabel('Speed')
plt.title('Recorded speeds 01.2016 - 05.2020')
plt.show()

plt.hist(avg_speeds_emp, bins=bind, density=True)
ylabels[15] = 0.15
plt.xticks(bind, xlabels)
plt.yticks(np.arange(0.0, 0.30, 0.01), ylabels)
plt.ylabel('Relative frequency')
plt.xlabel('Average speed')
plt.title('Recorded average speeds 01.2016 - 05.2020')
plt.show()


x_d = np.linspace(0, 40, 4000)
# instantiate and fit the KDE model
x = np.reshape(all_rec_speeds, (-1,1))
#params = {'bandwidth': np.logspace(-1, 1, 50)}
#grid = GridSearchCV(KernelDensity(), params)
#grid.fit(x)
#print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
kde = KernelDensity(bandwidth=0.85, kernel='gaussian')
kde.fit(x)
x.std()
print("standard deviation: ", x.std())
print("mean: ", np.mean(x))
print("median: ", np.median(x))
# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])
plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(all_rec_speeds, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.ylim(-0.02, 0.30)
plt.ylabel('Density')
plt.xlabel('Speed')
plt.title('Estimated pdf for recorded speeds, 01.2016 - 05.2020')
plt.yticks()
plt.show()

plt.hist(all_rec_speeds, bins=bind, density=True, alpha=0.5)
plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(all_rec_speeds, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.ylim(-0.02, 0.30)
plt.ylabel('Density')
plt.xlabel('Speed')
plt.title('Histogram and estimated pdf for speeds, 01.2016 - 05.2020')
plt.yticks()
plt.show()
#---------- end calc hist and pdf for speeds