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

def loading_joint_hierarchy( name_connection, name_length ):

    hierarchy = []
    length = []
    joint_hierarchy = []

    f_connection = open( name_connection )
    line = f_connection.readline() 
    while line:
        tmp_line = line.split( "," )
        tmp_line2 = tmp_line[1].split( "\n" )
        hierarchy.append( [int(tmp_line[0]),int(tmp_line2[0])] )
        line = f_connection.readline()
    f_connection.close

    f_length = open( name_length )
    line_length = f_length.readline() 
    while line_length:
        length.append( float(line_length) )
        line_length = f_length.readline()
    f_length.close

    for i in range( len( hierarchy ) ):
        joint_hierarchy.append( [hierarchy[i][0], hierarchy[i][1], length[i]] )

    return joint_hierarchy


def size_normalization( data, joint_heirarchy_length ):

    normalized_data = np.copy( data )

    for i, tmp_1 in enumerate( data ):
        for tmp_2 in joint_heirarchy_length:
            v = tmp_1[tmp_2[1]] - tmp_1[tmp_2[0]]
            v_ = v / np.linalg.norm(v)
            normalized_data[i][tmp_2[1]] = normalized_data[i][tmp_2[0]] + ( v_*tmp_2[2] )

    return normalized_data

def check_and_mkdir( path ):
    if not os.path.exists( path ):
        os.mkdir( path )

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument( "--target_dataset", help="augmentation target dataset name", default="original" )
    args = parser.parse_args()

    joint_heirarchy_length = loading_joint_hierarchy( "../joint_hierarchy.txt", "../joint_length.txt" )

    files = glob.glob( "../dataset/" + args.target_dataset + "/**/*.csv", recursive=True )
    for i,file_ in enumerate( files ):

        name = file_.split( '.csv' )[0]
        tmp = name.split( '/' )

        data, item = loading_file( name + ".csv" )
        data = np.array( data, dtype=float )
        
        normalized_data = size_normalization( data, joint_heirarchy_length )

        path = tmp[0] + '/' + tmp[1] + '/size_normalize'
        check_and_mkdir( path )
        path = path + '/' + tmp[3]
        check_and_mkdir( path )

        writing_file( normalized_data, item, path + '/' + tmp[4] + '_size_normalize.csv' )

if __name__ == '__main__':
    main()