
#
#  Who wrote this -- fix it yourself?
#  Date and Time?
#  Stuff at the header?
#  What course did you write this for?
#  What does this program do?
# 

import sys			# Why are you importing this?  What reason do you need this?
import datetime			# Why do you need this?  How is it used?
import numpy as np
import pandas
import math
import os
from sklearn.metrics.pairwise import euclidean_distances
import concurrent.futures


# A function to read the file and fill a dictionary with datapoint information:
#
# This should be given a file name and return a data structure.
# One might want to use pynema instead.
#

def get_datetime_obj(timestamp):
    timestamp_split = timestamp.split('.')[0]
    # Join every two characters with ':' (that is, get it into HH:MM:SS form), and then append the MS part.
    time_str = ':'.join([timestamp_split[index:index + 2] for index in range(0, len(timestamp_split), 2)]) + \
               ':' + timestamp.split('.')[1]
    # Use strptime to convert the string we just put together into a datetime object.
    date_time_obj = datetime.datetime.strptime(time_str, '%H:%M:%S:%f')
    return date_time_obj

def get_gps_data(gps_filename):
    gps_file = open(gps_filename, 'r')
    # Create an empty dictionary for the gps data.
    gps_info = {}
    # For every line in the gps data file.
    for line in gps_file:
        # If this line is an RMC line.
        if line.split(',')[0] == "$GPRMC":
            # Split the line on commas and set each value to a variable (based on its location per the protocol).
            try:
                label, timestamp, validity, latitude, lat_dir, longitude, long_dir, knots, course, \
                datestamp, variation, var_dir, checksum	= line.split(',')
            # If this line is missing a value or has too many, assume it's an error and skip the line.
            except ValueError:
                continue

            # If the validity is 'V' (per the protocol, a gps error), skip the line.
            if validity == 'V':
                continue

            # Get the timestamp into a datetime object.
            # Start by getting the HHMMSS part, the part before the '.'.
            timestamp_split	= timestamp.split('.')[0]
            # Join every two characters with ':' (that is, get it into HH:MM:SS form), and then append the MS part.
            time_str	= ':'.join([timestamp_split[index:index + 2] for index in range(0, len(timestamp_split), 2)]) + \
                       ':' + timestamp.split('.')[1]
            # Use strptime to convert the string we just put together into a datetime object.
            date_time_obj	= datetime.datetime.strptime(time_str, '%H:%M:%S:%f')

            # Convert the latitude into degrees.
            # This math might be wrong.
            degrees	= int(float(latitude) / 100)
            mins	= float(latitude) - (degrees * 100)
            # N is positive, S is negative.
            if lat_dir == 'N':
                fixed_latitude	= degrees + (mins / 60)
            else:
                fixed_latitude	= -degrees + (-mins / 60)

            # Convert the longitude into degrees.
            degrees	= int(float(longitude) / 100)
            mins	= float(longitude) - (degrees * 100)
            # E is positive, W is negative.
            if long_dir == 'E':
                fixed_longitude	= degrees + (mins / 60)
            else:
                fixed_longitude	= -degrees + (-mins / 60)

            # Add this line's information to the dictionary, with the timestamp as the key.
            # Using the timestamp means that multiple lines from the same time (e.g. if there's also GGA from this time)
            # will be combined.
            # If it's already in the dictionary, we want to update the entry instead of setting to keep from overriding
            #    any GGA information from that time (such as altitude).
            if timestamp in gps_info:
                gps_info[timestamp].update({'datetime': date_time_obj, 'latitude': round(fixed_latitude, 6),
                                            'longitude': round(fixed_longitude, 6), 'knots': float(knots),
                                            'course': course, 'variation': variation})
            else:
                gps_info[timestamp]	= {'datetime': date_time_obj, 'latitude': round(fixed_latitude, 6),
                                       'longitude': round(fixed_longitude, 6),
                                       'knots': knots, 'course': course, 'variation': variation}

        # If this line is a GGA line.
        elif line.split(',')[0] == "$GPGGA":
            # If the GPS quality indicator is not zero, this line should be valid.

                # Split the line on commas and set each value to a variable (based on its location per the protocol).
                try:
                    label, timestamp, latitude, lat_dir, longitude, long_dir, gps_quality, num_satellites, horiz_dilution, \
                    antenna_alt, antenna_alt_units, geoidal, geo_units, age_since_update, checksum	= line.split(',')
                # If this failed, that means something is wrong with this line.
                except ValueError:
                    continue
                if gps_quality != '0':
                # Convert the latitude into degrees.
                    degrees	= int(float(latitude) / 100)
                    mins	= float(latitude) - (degrees * 100)
                    # N is positive, S is negative.
                    if lat_dir == 'N':
                        fixed_latitude	= degrees + (mins / 60)
                    else:
                        fixed_latitude	= -degrees + (-mins / 60)

                    # Convert the longitude into degrees.
                    degrees	= int(float(longitude) / 100)
                    mins	= float(longitude) - (degrees * 100)
                    # E is positive, W is negative.
                    if long_dir == 'E':
                        fixed_longitude	= degrees + (mins / 60)
                    else:
                        fixed_longitude	= -degrees + (-mins / 60)

                    # Add this line's information to the dictionary, with the timestamp as the key.
                    # Using the timestamp means that multiple lines from the same time
                    # (e.g. if there's also RMC from this time) will be combined.
                    # If it's already in the dictionary, we want to update the entry instead of setting to keep
                    #   from overriding any RMC information from that time (such as course).
                    if timestamp in gps_info:
                        gps_info[timestamp].update({'latitude': round(fixed_latitude, 6),
                                                    'longitude': round(fixed_longitude, 6),
                                                    'altitude': float(antenna_alt)})
                    else:
                        gps_info[timestamp]	= {'latitude': round(fixed_latitude, 6), 'longitude': round(fixed_longitude, 6),
                                               'altitude': float(antenna_alt)}
        # We don't care about any other protocols.
        else:
            continue
    return gps_info


#
# A function to remove redundant entries from the dictionary (to lower the number of points while preserving the line).
#
# For example, if the car is traveling in a straight line, then
# delete the middle point.
# Or, maybe don't.
# Perhaps there is a bitter method.
def remove_redundant_GPS_points(gps_info:{}):
    # Get a list of all of the keys for the dictionary of GPS info.
    key_list	= list(gps_info.keys())
    # Iterate through the list of keys.
    for key_index in range(0, len(key_list) - 1):
        # Get the coordinates of the current and next entries in the dictionary.
        this_latitude   = gps_info[key_list[key_index]]['latitude']
        next_latitude	= gps_info[key_list[key_index + 1]]['latitude']
        this_longitude	= gps_info[key_list[key_index]]['longitude']
        next_longitude	= gps_info[key_list[key_index + 1]]['longitude']

        # If the coordinates are the same (rounded down a bit, to account for GPS skew)
        # meaning we haven't moved, remove one of the values.
        if round(this_latitude, 5) == round(next_latitude, 5) and round(this_longitude, 5) == round(next_longitude, 5):
            gps_info.pop(key_list[key_index])
            continue

        # If we didn't remove the data for the key right before this one, and this isn't the first key.
        if key_index != 0 and key_list[key_index - 1] in gps_info:

            # Get coordinates of the preceding entry.
            last_latitude 	= gps_info[key_list[key_index - 1]]['latitude']
            last_longitude 	= gps_info[key_list[key_index - 1]]['longitude']

            # Get the latitude and longitude differences in both directions.
            lat_diff_from_last	= abs(this_latitude - last_latitude)
            long_diff_from_last	= abs(this_longitude - last_longitude)
            lat_diff_from_next	= abs(this_latitude - next_latitude)
            long_diff_from_next	= abs(this_longitude - next_longitude)

            # If our speed is constant and we're moving in a straight line, remove the unnecessary point in between.
            tolerance	= 0.001   # This is WAY TOO BIG!!
            if abs(lat_diff_from_last - lat_diff_from_next) <= tolerance and \
                    abs(long_diff_from_last - long_diff_from_next) <= tolerance :
                gps_info.pop(key_list[key_index])
                continue

            # If the last entry has speed data (some lines may not if they were only GGA).
            if 'knots' in gps_info[key_list[key_index - 1]]:

                # Get the speed of the last entry.
                last_speed	= float(gps_info[key_list[key_index - 1]]['knots'])

                # Remove the entry if the difference in latitude or longitude is too high compared to the speed.
                # This manages errors where a GPS reading is way off (e.g. in the middle of the ocean.)
                if long_diff_from_last/10 > last_speed or lat_diff_from_next/10 > last_speed:
                    gps_info.pop(key_list[key_index])
                    continue

    return gps_info
#
# Write the kml file.
#
# REWRITE THIS TO TAKE IN A FILENAME.
#
# Write now it outputs a path.
# Instead we want to output a flag for where the car spent a lot of time.
#
def write_out_KML_file(kml_filename, gps_info):
    # Open it to write (such that it will be created if it does not already exist).
    kml_file	= open(kml_filename, 'w+')
    # Write the heading and style information.
    kml_file.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
    kml_file.write("<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n")
    kml_file.write("<Document>")
    # Every line is guaranteed to have a latitude and longitude.
    # If one does not have an altitude, it will just use the last altitude we had.
    last_alt	= 0
    # Iterate through all of the entries we didn't prune.
    for stop in gps_info:
        kml_file.write("<Placemark>\n")
        kml_file.write("<Style id='normalPlacemark'>\n")
        kml_file.write("<IconStyle>\n")
        kml_file.write("<Icon>\n")
        if stop['stop_time'] == 15:
            kml_file.write("<href>http://maps.google.com/mapfiles/kml/paddle/grn-circle.png</href>")
        elif stop['stop_time'] == 45:
            kml_file.write("<href>http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png</href>")
        kml_file.write("</Icon>\n</IconStyle>\n</Style>")
        kml_file.write("<Point>\n")
        kml_file.write("<coordinates>" + str(stop['longitude']) + ',' + str(stop['latitude']) + ',' + str(last_alt) + "</coordinates>\n")
        kml_file.write("</Point>\n")
        kml_file.write("    </Placemark>\n")
    kml_file.write("</Document>")
    kml_file.write("</kml>\n")

def create_df(gps_info):
    dict_list = []
    for key in gps_info.keys():
        if "knots" not in gps_info[key] or gps_info[key]['knots'] == 0:
            temp_dict = {}
            temp_dict['c_latitude'] = gps_info[key]['latitude']
            temp_dict['c_longitude'] = gps_info[key]['longitude']
            temp_dict['m_latitude'] = gps_info[key]['latitude']
            temp_dict['m_longitude'] = gps_info[key]['longitude']
            temp_dict['timestamp'] = get_datetime_obj(key)
            temp_dict['time_diff'] = 0
            dict_list.append(temp_dict)
    df = pandas.DataFrame(dict_list)
    # df.drop(df[['c_latitude', 'c_longitude']].round(2).duplicated().loc[lambda x: x].index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    # print(df)
    return df

def check_if_valid_stop(col_dict, row_dict):
    time_diff = col_dict['time_diff']
    time_in_minutes = time_diff//60
    tolerance = 0.001

    if time_in_minutes >= 45:
        return 45
    elif time_in_minutes >= 15:
        return 15
    return 0

def calculate_distance(point, c_latitude, c_longitude):

    latitude = math.pow(point['m_latitude'] - c_latitude, 2)
    longitude = math.pow(point['m_longitude'] - c_longitude, 2)
    distance = math.sqrt(latitude + longitude)
    return distance

def agglomeration(df:pandas.DataFrame):
    """
    Performs hierarchial agglomeration on the given dataframe
    :param df:
    :return:
    """

    cluster_len = len(df)
    stops = []
    #initializing the dictionaries with keys
    print("Df Len : " + str(len(df)))
    while cluster_len >= 0 and len(df) >= 2:

        data = df.drop(["timestamp", "time_diff","c_latitude","c_longitude"], axis=1)

        eucliean_distance_matrix = np.tril(euclidean_distances(data, data))
        # finding the index of minimum non-zero element in euclidean distance matrix, which will be merged in this iteration.
        min_val_arr = np.where(eucliean_distance_matrix == np.min(eucliean_distance_matrix[np.nonzero(eucliean_distance_matrix)]))
        min_index_row,min_index_col = min_val_arr[0][0].item(),min_val_arr[1][0].item()
        time_stamp = df.loc[min_index_col,'timestamp']
        # temp_df = pandas.merge(df.loc[min_index_col], df.loc[min_index_row])
        c_latitude = (df.at[min_index_col, 'c_latitude'] + df.at[min_index_row, 'c_latitude'])/2
        c_longitude = (df.at[min_index_col, 'c_longitude'] + df.at[min_index_row, 'c_longitude'])/2
        df.at[min_index_col, 'c_latitude'] = round(c_latitude, 6)
        df.at[min_index_col, 'c_longitude'] = round(c_longitude, 6)

        if calculate_distance(df.loc[min_index_col], c_latitude,c_longitude) > calculate_distance(df.loc[min_index_row],c_latitude,c_longitude):
            df.at[min_index_col, 'm_latitude'] = df.at[min_index_row, 'm_latitude']
            df.at[min_index_col, 'm_longitude'] = df.at[min_index_row,'m_longitude']
        if df.loc[min_index_col, 'timestamp'] > df.loc[min_index_row, 'timestamp']:
            df.loc[min_index_col, 'timestamp'] = df.loc[min_index_row, 'timestamp']
            df.loc[min_index_col, 'time_diff'] = (time_stamp - df.loc[min_index_row, 'timestamp']).seconds
        else:
            df.loc[min_index_col, 'timestamp'] = time_stamp
            df.loc[min_index_col, 'time_diff'] = (df.loc[min_index_row, 'timestamp'] - time_stamp).seconds
        stop_time = check_if_valid_stop(df.loc[min_index_col], df.loc[min_index_row])
        if stop_time != 0:
            temp = {
                    "latitude": df.loc[min_index_col, 'm_latitude'],
                    "longitude": df.loc[min_index_col, 'm_longitude'],
                    "stop_time": stop_time
                    }
            stops.append(temp)

        df.drop(inplace=True, index=min_index_row)
        df.reset_index(inplace=True, drop=True)
        cluster_len -= 1
    stops_df = remove_redundant_stops(stops)
    return stops_df
#
#  Here is the main routine:
#

def remove_redundant_stops(stops):
    stop_df = pandas.DataFrame(stops)
    if len(stops)> 0 :

        stop_df.sort_values(by=['latitude', 'longitude'], inplace=True)
        stop_df.drop(stop_df[['latitude', 'longitude']].round(2).duplicated().loc[lambda x: x].index, inplace=True)
        # print(stop_df)
    return stop_df

def get_stops_from_file(gps_filename):
    gps_info = get_gps_data(gps_filename)
    cleaned_gps_info = remove_redundant_GPS_points(gps_info)
    df = create_df(cleaned_gps_info)
    stops_df = agglomeration(df)
    return stops_df

def main( ):
    # Call the functions to run the program overall.

    gps_filename = sys.argv[1]
    kml_filename = sys.argv[2]
    gps_dir = "/Users/mayur/CSCI_720/csci_720/BDA_Project/"
    future_list = []
    all_stops = pandas.DataFrame()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for subdir, dirs, files in os.walk(gps_dir):
            for file in files:
                if file.startswith('2022_') and file.endswith(".txt"):
                    gps_filename = os.path.join(subdir, file)
                    print(gps_filename)
                    # Each file is submitted to a thread, to read and process the data. After submitting the task we add
                    # the thread handler to list.
                    future = executor.submit(get_stops_from_file, gps_filename)
                    future_list.append(future)
    for future in concurrent.futures.as_completed(future_list):
        file_stops = future.result()
        if len(file_stops) > 0 :
            all_stops = all_stops.append(file_stops)

    # all_stops.drop(all_stops[['latitude', 'longitude']].round(2).duplicated().loc[lambda x: x].index, inplace=True)
    print(len(all_stops))
    write_out_KML_file(kml_filename, all_stops.to_dict(orient="records"))



if ( __name__ == "__main__" ) :
    main()

