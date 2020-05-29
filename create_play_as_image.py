import ngs_player_tracking as ngs
import argparse
import datetime
import time
import glob
import json
import os, sys
import numpy as np
import pandas as pd
#from sklearn.preprocessing import KBinsDiscretizer
#from sklearn.cluster import MeanShift
import ruptures as rpt

from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, Panel, LinearColorMapper, BasicTicker, ColorBar, PrintfTickFormatter
from bokeh.layouts import column, row, gridplot, widgetbox
from bokeh.models.widgets import Tabs
from bokeh.transform import transform
from bokeh.palettes import Viridis256, Reds8, Blues8, Plasma256
from bokeh.io import export_png

field_x_meters = 120.0  # Actually Yards
field_y_meters = 53.33 # Actually Yards
ex_play_time_start = '22:21:47.7' # Time of hike (visually approximated using animation app)
ex_play_time_end = '22:21:54.0' # Time of dead ball tackle (visually approximated using animation app)
ball_speed_min = 0
ball_speed_max = 45

def argparser():
	parser = argparse.ArgumentParser()

	req_args = parser.add_argument_group('required arguments')
	#req_args.add_argument('-f', dest='data_files', required=True, nargs='*', help='The path and filename for the data(s) to plot.')
	opt_args = parser.add_argument_group('optional arguments')
	#opt_args.add_argument('-v', dest='video', required=False, help='The path to the images to make the video.')

	args = parser.parse_args()
	return args

def bokeh_team_colors(player, team):
	""" Set the team colors"""
	if player=='ball':
		return '#AB3424'
	elif team=='home':
		return '#E3E3E3'
	elif team=='away':
		return '#404040'

	return '#FFFFFF'

def bokeh_time_colors(times, color_range):
	"""
	Take the list of times and color range, and return a list of colors that properly maps them
	"""

	print (times)
	print (color_range)

	return

def convert_time_to_seconds(time_str):
	""" Convert string time HH:MM:SS.ms to a seconds float """
	return 3600*int(time_str[0:2])+60*int(time_str[3:5])+int(time_str[6:8])+float(time_str[8:])

def get_velocity_list(positions, timesteps, units='m/s'):
  """
  Calculate a velocity given the list of xy as [(x0,y0),(x1,y1),..] and timesteps as [t0,t1,t2]
  velocity at t0 will be 0
  """

  velocity_list = []
  velocity_list.append((0,0))
  for i in range(len(positions) - 1):
  	pos1 = positions[i]
  	pos2 = positions[i+1]
  	v = get_velocity(pos1, pos2, timesteps[i], timesteps[i+1])
  	velocity_list.append(v)

  return velocity_list

def get_velocity(pos_1, pos_2, time_1, time_2, units='m/s'):
	"""
	Calculate a velocity given two positions
	"""

	#vx = field_x_meters * float(pos_2[0] - pos_1[0]) / float(abs(time_2 - time_1))
	#vy = field_x_meters * float(pos_2[1] - pos_1[1]) / float(abs(time_2 - time_1))
	if (time_2==time_1): # Saw this as an issue in this json file: D:/Sports Analytics/sportradar/eagles-bucs-week2/0307ce31-aa17-4985-a0c4-9a01f1783af9.json
		# Just a potential for some entries at the same time. The positions here are also the same, so zero velocity makes sense
		return (0, 0)
	else:
		vx = float(pos_2[0] - pos_1[0]) / float(abs(time_2 - time_1))
		vy = float(pos_2[1] - pos_1[1]) / float(abs(time_2 - time_1))

	return (vx, vy)

def get_ball_velocity(json_file_path):
	"""
	Using this to test out a methodology for programatically getting hike and dead ball time marks
	In theory, the ball should be stationary at the hike, and maybe it reaches another stopped point at, or nearly after the dead ball whistle
	"""
	#print (json_file_path)
	df_all_tracking, times = ngs.extract_player_tracking_data(json_file_path)
	#print ('  ',list(df_all_tracking['description'])[0])
	df_filtered = df_all_tracking[df_all_tracking['player_id']=='ball'].sort_values(by='time')
	df_filtered['time_val'] = df_filtered['time'].apply(lambda x: convert_time_to_seconds(x))

	all_velocities = {}
	xpos = np.array(df_filtered['x'])
	ypos = np.array(df_filtered['y'])
	positions = list(zip(xpos, ypos))
	timesteps = np.array(df_filtered['time_val'])
	velocities = get_velocity_list(positions, timesteps)
	indices = list(df_filtered.index)
	all_velocities = {**all_velocities, **dict(zip(indices, velocities))}

	#  Compile the velocity data structures of all players and add it to the dataframe
	df_filtered['velocity'] = df_filtered.index.to_series().map(all_velocities)
	df_filtered['speed'] = df_filtered['velocity'].apply(lambda x: (x[0]**2 + x[1]**2)**0.5)
	
	#print (min(df_filtered['speed']), '\t', max(df_filtered['speed']))
	# ball speed ranged from 0 to 75.1, and [0,45] is ~90% of the data values
	#  so going to make 45 the max and 0 the min for the color mapper

	#print (df_filtered.head(100))
	#df_filtered.to_csv('./ball_velocity.csv')

	## Using sklearn discretizer
	#enc = KBinsDiscretizer(n_bins=10, encode='onehot')
	#ball_speed_binned = enc.fit_transform(np.array(list(df_filtered['speed'])).reshape(-1, 1))
	#print (ball_speed_binned)

	## Using sklearn.cluster.MeanShift (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html)
	#clustering = MeanShift().fit()

	## Using ruptures library
	speed_signal = np.array(df_filtered['speed']).reshape(-1,1)
	speed_breakpts = 2
	speed_detection = rpt.Pelt(model="rbf").fit(speed_signal)
	speed_result = speed_detection.predict(pen=10)

	return speed_result[:-1], [list(df_filtered['time'])[x] for x in speed_result[:-1]], list(df_filtered['time'])[-1]

def bokeh_plot_players(existing_plot, df_player_data):
	"""
	Given an existing bokeh plot, add on the players, ball, and velocity arrows as glyphs on top of whatever is there already
	"""

	# Convert x/y to meters
	#df_player_data['x'] = field_x_meters * df_player_data['x'] # Already in yards
	#df_player_data['y'] = field_y_meters * df_player_data['y'] # Already in yards
	#df_player_data['color'] = df_player_data['team'].apply(lambda x: bokeh_team_colors(x))

	# Create the speed & velocity fields
	#print (df_player_data.columns)
	players = list(set(df_player_data['player_id']))
	df_player_data['time_val'] = df_player_data['time'].apply(lambda x: convert_time_to_seconds(x))

	# [x] Calculate the velocity for this data frame
	# [x] Calculate angle of velocity
	all_velocities = {}
	for p in players:
		#   Get sorted df/list of the xy positions by time
		df_player = df_player_data[df_player_data['player_id']==p].sort_values(by=['time'])
		#print (df_player.head(5))
		xpos = np.array(df_player['x'])
		ypos = np.array(df_player['y'])
		positions = list(zip(xpos, ypos))
		timesteps = np.array(df_player['time_val'])
		velocities = get_velocity_list(positions, timesteps)
		indices = list(df_player.index)

		#   Return a dictionary (???) of the time and player with the velocity as the value
		all_velocities = {**all_velocities, **dict(zip(indices, velocities))}

	#  Compile the velocity data structures of all players and add it to the dataframe
	df_player_data['velocity'] = df_player_data.index.to_series().map(all_velocities)
	df_player_data['speed'] = df_player_data['velocity'].apply(lambda x: (x[0]**2 + x[1]**2)**0.5)
	
	df_player_data['time_val'] = df_player_data.apply(lambda x: 3600*int(x['time'][0:2])+60*int(x['time'][3:5])+int(x['time'][6:8])+float(x['time'][8:]), axis=1)
	df_player_data['color'] = df_player_data.apply(lambda x: bokeh_team_colors(x['player_id'], x['team']), axis=1)
	source = ColumnDataSource(df_player_data)

	# Create the schema for the velocity arrows/lines
	"""
	df_player_data['x_plus_vel'] = df_player_data.apply(lambda i: i['x'] + i['velocity'][0], axis=1)
	df_player_data['y_plus_vel'] = df_player_data.apply(lambda i: i['y'] + i['velocity'][1], axis=1)
	x_start = [i for i in list(df_player_data[df_player_data['player']!='Ball']['x'])]
	y_start = [i for i in list(df_player_data[df_player_data['player']!='Ball']['y'])]
	x_end = [i for i in list(df_player_data[df_player_data['player']!='Ball']['x_plus_vel'])]
	y_end = [i for i in list(df_player_data[df_player_data['player']!='Ball']['y_plus_vel'])]
	x_list = [[x_start[i], x_end[i]] for i in range(len(x_start))]
	y_list = [[y_start[i], y_end[i]] for i in range(len(y_start))]
	#print (x_list)
	#print (y_list)
	#print (df_player_data.head(10))"""

	## Create the multi line dataset for the player tracks
	x_list_h = []
	x_list_h_adj = []
	y_list_h = []
	y_list_h_adj = []
	x_list_a = []
	x_list_a_adj = []
	y_list_a = []
	y_list_a_adj = []
	x_list_b = []
	x_list_b_adj = []
	y_list_b = []
	y_list_b_adj = []
	speed_list_b = []
	for player in list(set(list(df_player_data['player_id']))):
		df_filtered = df_player_data[df_player_data['player_id']==player]
		df_filtered = df_filtered.sort_values(by='time')
		team_filtered = list(df_filtered['team'])[0]
		if team_filtered=='home':
			x_list_h.append(list(df_filtered['x']))
			x_h_adj = list(df_filtered['x'])
			x_list_h_adj.append([[x_h_adj[i], x_h_adj[i+1]] for i in range(len(x_h_adj)-1)])
			y_list_h.append(list(df_filtered['y']))
			y_h_adj = list(df_filtered['y'])
			y_list_h_adj.append([[y_h_adj[i], y_h_adj[i+1]] for i in range(len(y_h_adj)-1)])
		elif team_filtered=='away':
			x_list_a.append(list(df_filtered['x']))
			x_a_adj = list(df_filtered['x'])
			x_list_a_adj.append([[x_a_adj[i], x_a_adj[i+1]] for i in range(len(x_a_adj)-1)])
			y_list_a.append(list(df_filtered['y']))
			y_a_adj = list(df_filtered['y'])
			y_list_a_adj.append([[y_a_adj[i], y_a_adj[i+1]] for i in range(len(y_a_adj)-1)])
		elif team_filtered=='ball':
			x_list_b.append(list(df_filtered['x']))
			x_b_adj = list(df_filtered['x'])
			x_list_b_adj.append([[x_b_adj[i], x_b_adj[i+1]] for i in range(len(x_b_adj)-1)])
			y_list_b.append(list(df_filtered['y']))
			y_b_adj = list(df_filtered['y'])
			y_list_b_adj.append([[y_b_adj[i], y_b_adj[i+1]] for i in range(len(y_b_adj)-1)])
			speed_list_b.append(list(df_filtered['speed']))

	## Multi-line adjusted for color spectrum
	#x_list_h_adj = [[x_list_h[i], x_list_h[i+1]] for i in range(len(x_list_h))]
	#print (len(x_list_h_adj), len(x_list_a_adj), len(x_list_b_adj))

	## Create the starting position and ending position datasets
	x_start_h = [i[0] for i in x_list_h]
	y_start_h = [i[0] for i in y_list_h]
	x_start_a = [i[0] for i in x_list_a]
	y_start_a = [i[0] for i in y_list_a]
	x_start_b = [i[0] for i in x_list_b]
	y_start_b = [i[0] for i in y_list_b]

	"""
	# Create the soccer paint lines (6, 18, midfield)
	existing_plot.line([105.0/2.0,105.0/2.0],[0,68],color='#FFFFFF', line_width=1)
	existing_plot.line([0,6,6,0],[24,24,44,44],color='#FFFFFF', line_width=1)
	existing_plot.line([105,99,99,105],[24,24,44,44],color='#FFFFFF', line_width=1)
	existing_plot.line([0,18,18,0],[12,12,56,56],color='#FFFFFF', line_width=1)
	existing_plot.line([105,87,87,105],[12,12,56,56],color='#FFFFFF', line_width=1)

	# Create the Goal Marks
	existing_plot.line([0,1],[30.34,30.34],color='#FFFFFF',line_width=1)
	existing_plot.line([0,1],[37.66,37.66],color='#FFFFFF',line_width=1)
	existing_plot.line([105,104],[30.34,30.34],color='#FFFFFF',line_width=1)
	existing_plot.line([105,104],[37.66,37.66],color='#FFFFFF',line_width=1)
	"""

	## Create the list of colors for dark --> light time progression
	times = sorted(list(set(list(df_player_data['time_val']))))
	mapper_h = LinearColorMapper(palette=Viridis256, low=times[0], high=times[-1])
	mapper_a = LinearColorMapper(palette=Blues8, low=times[0], high=times[-1])
	mapper_speed = LinearColorMapper(palette=Plasma256, low=ball_speed_min, high=ball_speed_max) 

	"""
	for i in range(len(x_list_h_adj)):
		source = ColumnDataSource(data={
			'x': x_list_h_adj[i],
			'y': y_list_h_adj[i],
			'time': times,
		})
		existing_plot.multi_line(xs='x', ys='y', color=transform('time',mapper_h), line_width=3, source=source)

	for i in range(len(x_list_a_adj)):
		source = ColumnDataSource(data={
			'x': x_list_a_adj[i],
			'y': y_list_a_adj[i],
			'time': times,
		})
		existing_plot.multi_line(xs='x', ys='y', color=transform('time',mapper_a), line_width=3, source=source)	
	"""
	for i in range(len(speed_list_b)):
		source = ColumnDataSource(data={
			'x': x_list_b_adj[i],
			'y': y_list_b_adj[i],
			'speed': speed_list_b[i][:-1],
		})
		existing_plot.multi_line(xs='x', ys='y', color=transform('speed',mapper_speed), line_width=3, source=source)


	#existing_plot.multi_line(x_list_h, y_list_h, color='#F54538', line_width=1)
	#existing_plot.multi_line(x_list_a, y_list_a, color='#47FFF5', line_width=1) #47FFF5 #4755F5
	#existing_plot.multi_line(x_list_b, y_list_b, color='#FFFFFF', line_width=3)
	
	#existing_plot.circle(x=x_start_h, y=y_start_h, color='#F54538', size=5)
	#existing_plot.circle(x=x_start_a, y=y_start_a, color='#47FFF5', size=5)
	#existing_plot.circle(x=x_start_b, y=y_start_b, color='#FFFFFF', size=5)

	return existing_plot

def bokeh_create_play_image(json_file_path, export_name, play_time_start, play_time_end):
	"""
	Make the player tracking image
	"""

	df_all_tracking, times = ngs.extract_player_tracking_data(json_file_path)

	# Cut down df_all_tracking to just the time between start/end, the "live ball" time
	df_all_tracking = df_all_tracking[df_all_tracking['time'] >= play_time_start]
	df_all_tracking = df_all_tracking[df_all_tracking['time'] <= play_time_end]

	#play_plot = figure(title='',plot_width=int(field_x_meters*10),plot_height=int(field_y_meters*10),x_range=(0,field_x_meters),y_range=(0,field_y_meters))
	play_plot = figure(title='',plot_width=224,plot_height=224,x_range=(0,field_x_meters),y_range=(0,field_y_meters)) # 224x224 is ideal for CV NNs
	play_plot.background_fill_color = '#404040'
	play_plot.grid.grid_line_color = None
	play_plot.axis.axis_line_color = None
	play_plot.axis.major_tick_line_color = None

	play_plot = bokeh_plot_players(play_plot, df_all_tracking)
	play_plot.toolbar.logo = None
	play_plot.toolbar_location = None
	play_plot.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
	play_plot.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
	play_plot.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
	play_plot.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
	#play_plot.xaxis.major_label_text_font_size = '0pt'  # turn off x-axis tick labels
	#play_plot.yaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels
	play_plot.xaxis.major_label_text_color = None  # turn off x-axis tick labels leaving space
	play_plot.yaxis.major_label_text_color = None  # turn off y-axis tick labels leaving space

	export_png(play_plot, filename=export_name)

	return

def main():
	"""
	Create a play image for everything in the external drive folder
	"""

	## Next, loop through all of the play files in the 5 game subfolders here: D:\Sports Analytics\sportradar\
	#    and get a set of all the play_tracking>play>play_type string values. I think we'll only want the pass or rush ones
	# {'kickoff', 'punt', 'field_goal', 'two_point_conversion', 'pass', 'sack', 'extra_point', 'unknown', 'rush'}
	#  Then save each png plot image with a run/pass prefix and the ending being the play_id number/string (might need to be a number)
	#  Save all of them in the same directory
	path_play_files = 'D:/Sports Analytics/sportradar/'
	game_folders = ['eagles-bucs-week2','jaguars-patriots-week2','rams-raiders-week1','ravens-bengals-week2','redskins-colts-week2']
	#game_folders = ['ravens-bengals-week2']
	## play_types = []
	counter_pass = 1
	counter_rush = 1
	all_returned_breaks = []
	for game in game_folders:
		print (game)
		play_jsons = glob.glob(path_play_files + game + '/*.json')
		for p_json in play_jsons:
			play_json_data = json.loads(open(p_json).read())
			play_type = play_json_data['play_tracking']['play']['play_type']
			#play_types.append(play_type)
			if play_type in ['pass', 'rush']:
				#print (p_json)
				break_idx, break_times, last_time = get_ball_velocity(p_json)
				if len(break_times)==0:
					#print ('  0 break_times:', p_json)
					continue
				play_time_start = break_times[0]
				if len(break_times)==1:
					play_time_end = last_time
				elif len(break_times)==2:
					play_time_end = break_times[1]
				elif len(break_times)>=3:
					play_time_end = break_times[2]

				if play_type=='pass':
					bokeh_create_play_image(p_json, './play_images_ballspeed/' + play_type + '_' + str(counter_pass) + '.png', play_time_start, play_time_end)
					counter_pass += 1
				if play_type=='rush':
					bokeh_create_play_image(p_json, './play_images_ballspeed/' + play_type + '_' + str(counter_rush) + '.png', play_time_start, play_time_end)
					counter_rush += 1

			#if play_type=='pass':
			#	#bokeh_create_play_image(p_json, 'play_images/' + play_type + '_' + str(counter_pass) + '.png')
			#	#counter_pass += 1
			#	break_idx, break_times = get_ball_velocity(p_json)
			#	all_returned_breaks.append(break_idx)
			#	if len(break_idx) == 5:
			#		print ('  pass', p_json, play_json_data['play_tracking']['play']['description'], break_times)
			#	#sys.exit(1)
			#elif play_type=='rush':
			#	#bokeh_create_play_image(p_json, 'play_images/' + play_type + '_' + str(counter_pass) + '.png')
			#	#counter_rush += 1
			#	break_idx, break_times = get_ball_velocity(p_json)
			#	all_returned_breaks.append(break_idx)
			#	if len(break_idx) == 5:
			#		print ('  rush', p_json, play_json_data['play_tracking']['play']['description'], break_times)
			#	#sys.exit(1)

	"""
	print ('Max Breaks:',max([len(x) for x in all_returned_breaks]))
	print ('1 Break(s):',len([x for x in all_returned_breaks if len(x)==1]))
	print ('2 Break(s):',len([x for x in all_returned_breaks if len(x)==2]))
	print ('3 Break(s):',len([x for x in all_returned_breaks if len(x)==3]))
	print ('4 Break(s):',len([x for x in all_returned_breaks if len(x)==4]))
	print ('5 Break(s):',len([x for x in all_returned_breaks if len(x)==5]))
	print ('6 Break(s):',len([x for x in all_returned_breaks if len(x)==6]))
	print ('7 Break(s):',len([x for x in all_returned_breaks if len(x)==7]))
	"""

	return

def main_testing():
	"""
	Trying to create an image from the data of a single tracked play
	"""

	ex_play = 'example_play.json'
	df_all_tracking, times = ngs.extract_player_tracking_data(ex_play)

	# Cut down df_all_tracking to just the time between start/end, the "live ball" time
	df_all_tracking = df_all_tracking[df_all_tracking['time'] >= ex_play_time_start]
	df_all_tracking = df_all_tracking[df_all_tracking['time'] <= ex_play_time_end]
	
	#print (df_all_tracking.head(20))
	#print (set(list(df_all_tracking['team'])))
	#print (set(list(df_all_tracking['player_id'])))

	play_plot = figure(title='',plot_width=int(field_x_meters*10),plot_height=int(field_y_meters*10),x_range=(0,field_x_meters),y_range=(0,field_y_meters))
	play_plot.background_fill_color = '#404040'
	play_plot.grid.grid_line_color = None
	play_plot.axis.axis_line_color = None
	play_plot.axis.major_tick_line_color = None


	"""
	Leverage *maybe* existing code to plot the play using js, or plotly, or matplotlib, or bokeh or whatever
		have the player locations and the tracking lines all on the plot, with a black background
		then just save that plot as an image
	"""
	play_plot = bokeh_plot_players(play_plot, df_all_tracking)
	play_plot.toolbar.logo = None
	play_plot.toolbar_location = None
	play_plot.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
	play_plot.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
	play_plot.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
	play_plot.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
	#play_plot.xaxis.major_label_text_font_size = '0pt'  # turn off x-axis tick labels
	#play_plot.yaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels
	play_plot.xaxis.major_label_text_color = None  # turn off x-axis tick labels leaving space
	play_plot.yaxis.major_label_text_color = None  # turn off y-axis tick labels leaving space


	export_png(play_plot, filename='play_images/example_play.png')

	output_file('nfl_play_image.html')
	save(play_plot)


	## Next, loop through all of the play files in the 5 game subfolders here: D:\Sports Analytics\sportradar\
	#    and get a set of all the play_tracking>play>play_type string values. I think we'll only want the pass or rush ones
	# {'kickoff', 'punt', 'field_goal', 'two_point_conversion', 'pass', 'sack', 'extra_point', 'unknown', 'rush'}
	#  Then save each png plot image with a run/pass prefix and the ending being the play_id number/string (might need to be a number)
	#  Save all of them in the same directory
	path_play_files = 'D:/Sports Analytics/sportradar/'
	game_folders = ['eagles-bucs-week2','jaguars-patriots-week2','rams-raiders-week1','ravens-bengals-week2','redskins-colts-week2']
	play_types = []
	for game in game_folders:
		play_jsons = glob.glob(path_play_files + game + '/*.json')
		for p_json in play_jsons:
			play_json_data = json.loads(open(p_json).read())
			play_type = play_json_data['play_tracking']['play']['play_type']
			play_types.append(play_type)

	play_types = set(play_types)
	print (play_types)


	return

if __name__ == '__main__':
	args = argparser()
	main()
	print ('Success!!')