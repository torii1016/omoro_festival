import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import argparse
import glob
import os

def loading_file( name ):

    data = np.empty( (0,26,3) , float )
    spilit_list = [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75]
    with open( name ) as f:
        reader = csv.reader(f)
        for j, row in enumerate( reader ):
            if j!=0:
                tmp = np.split(row,spilit_list)
                data = np.append( data, [tmp], axis=0 )
            else:
                item = row

    return data, item

def writing_file( data, item , name ):

    data = np.reshape( data, ( data.shape[0],-1 ) )
    with open( name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(item)
        for i in data:
            writer.writerow( i.tolist() )

def calculate_xyz_max_min( data ):

    x_max = y_max = z_max = -99999
    x_min = y_min = z_min = 99999

    for i in range( data.shape[0] ):
        tmp_x_max = np.max( data[i,:,0] )
        tmp_x_min = np.min( data[i,:,0] )
        tmp_y_max = np.max( data[i,:,1] )
        tmp_y_min = np.min( data[i,:,1] )
        tmp_z_max = np.max( data[i,:,2] )
        tmp_z_min = np.min( data[i,:,2] )

        if x_max<tmp_x_max:
            x_max = tmp_x_max
        if tmp_x_min<x_min:
            x_min = tmp_x_min
        if y_max<tmp_y_max:
            y_max = tmp_y_max
        if tmp_y_min<y_min:
            y_min = tmp_y_min
        if z_max<tmp_z_max:
            z_max = tmp_z_max
        if tmp_z_min<z_min:
            z_min = tmp_z_min
    
    return x_min, x_max, y_min, y_max, z_min, z_max

def normalization( data, min_v, max_v ):

    max_min = max_v-min_v
    mins = np.reshape( min_v.tolist()*data.shape[0]*data.shape[1], data.shape )

    normalized_data = ( data - mins )/max_min

    return normalized_data

def check_and_mkdir( path ):
    if not os.path.exists( path ):
        os.mkdir( path )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument( "--target_dataset", help="augmentation target dataset name", default="original" )
    args = parser.parse_args()

    files = glob.glob( "../dataset/" + args.target_dataset + "/**/*.csv", recursive=True )

    x_max = y_max = z_max = -99999
    x_min = y_min = z_min = 99999
    num_min = 9999
    num_max = -9999

    for file_ in files:

        name = file_.split( '.csv' )[0]
        tmp = name.split( '/' )
        print( tmp[4] )

        data, item = loading_file( name + ".csv" )
        data = np.array( data, dtype=float )

        tmp_x_min, tmp_x_max, tmp_y_min, tmp_y_max, tmp_z_min, tmp_z_max = calculate_xyz_max_min( data )

        if x_max<tmp_x_max:
            x_max = tmp_x_max
        if tmp_x_min<x_min:
            x_min = tmp_x_min
        if y_max<tmp_y_max:
            y_max = tmp_y_max
        if tmp_y_min<y_min:
            y_min = tmp_y_min
        if z_max<tmp_z_max:
            z_max = tmp_z_max
        if tmp_z_min<z_min:
            z_min = tmp_z_min

        if data.shape[0]<num_min:
            num_min = data.shape[0]
        if num_max<data.shape[0]:
            num_max = data.shape[0]

    min_v = np.array( [x_min,y_min,z_min] )
    max_v = np.array( [x_max,y_max,z_max] )
    print( x_min, x_max, y_min, y_max, z_min, z_max )
    print( "num_min: {}, num_max:{}".format( num_min, num_max ) )

    """
    files = glob.glob( "../dataset/" + args.target_dataset + "/**/*.csv", recursive=True )
    for file_ in files:

        name = file_.split( '.csv' )[0]
        tmp = name.split( '/' )

        data, item = loading_file( name + ".csv" )
        data = np.array( data, dtype=float )

        augmentated_data = normalization( data, min_v, max_v )

        path = tmp[0] + '/' + tmp[1] + '/normalize'
        check_and_mkdir( path )
        path = path + '/' + tmp[3]
        check_and_mkdir( path )

        writing_file( augmentated_data, item, path + '/' + tmp[4] + '_normalize.csv' )
    """

if __name__ == '__main__':
    main()