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


def augmentation_noise( data ):

    noise = np.random.normal( 0, 5, data.shape )

    augmentated_data = data + noise

    return augmentated_data

def augmentation_z_scaling( data ):

    tmp = [0,0,np.random.randint( -100, 100 )]*26
    scaling = np.empty( (0,26,3) , float )
    scaling = np.append( scaling, [np.array(tmp).reshape(26,3)], axis=0 )

    augmentated_data = data + scaling

    return augmentated_data

def augmentation_x_translation( data ):

    tmp = [np.random.randint( -100, 100 ),0,0]*26
    translation = np.empty( (0,26,3) , float )
    translation = np.append( translation, [np.array(tmp).reshape(26,3)], axis=0 )

    augmentated_data = data + translation

    return augmentated_data


def augmentation_y_translation( data ):

    tmp = [0, np.random.randint( -30, 30 ),0]*26
    translation = np.empty( (0,26,3) , float )
    translation = np.append( translation, [np.array(tmp).reshape(26,3)], axis=0 )

    augmentated_data = data + translation

    return augmentated_data



def augmentation_y_rotation( data ):

    augmentated_data = np.copy( data )

    theta = ( np.pi/4.0*2.0 )*np.random.rand() - np.pi/4.0 
    rotation = np.array([[np.cos(theta), 0, -np.sin(theta)],[0, 1, 0],[np.sin(theta), 0, np.cos(theta)]])
    for k, tmp_1 in enumerate( data ):
        for j, tmp_2 in enumerate( tmp_1 ):
            augmentated_data[k,j,:] = np.dot( rotation, data[k,j,:] )

    return augmentated_data

def check_and_mkdir( path ):
    if not os.path.exists( path ):
        os.mkdir( path )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument( "--data_name", help="csv file name", default="default" )
    parser.add_argument( "--aug_mode", help="augmentation mode, add noise[0] / z-value scaling[1] / x-value translation[2] / y-axis rotation[3] / y-value translation[4]", default=0, type=int )
    parser.add_argument( "--target_dataset", help="augmentation target dataset name", default="original" )
    args = parser.parse_args()

    files = glob.glob( "../dataset/" + args.target_dataset + "/**/*.csv", recursive=True )

    """
    data, item = loading_file( args.data_name + ".csv" )
    data = np.array( data, dtype=float )
    augmentated_data = augmentation_y_rotation( data )
    writing_file( augmentated_data, item, 'test.csv' )
    """

    if args.aug_mode==0:
        augmentation_time = 50
    else:
        augmentation_time = 30

    for file_ in files:

        name = file_.split( '.csv' )[0]
        tmp = name.split( '/' )

        data, item = loading_file( name + ".csv" )
        data = np.array( data, dtype=float )

        for i in range( augmentation_time ):
            if args.aug_mode==0:
                augmentated_data = augmentation_noise( data )

                path = tmp[0] + '/' + tmp[1] + '/noise'
                check_and_mkdir( path )
                path = path + '/' + tmp[3]
                check_and_mkdir( path )

                writing_file( augmentated_data, item, path + '/' + tmp[4] + '_noise_' + str( i+1 ) + '.csv' )

            elif args.aug_mode==1:
                augmentated_data = augmentation_z_scaling( data )

                path = tmp[0] + '/' + tmp[1] + '/scaling'
                check_and_mkdir( path )
                path = path + '/' + tmp[3]
                check_and_mkdir( path )

                writing_file( augmentated_data, item, path + '/' + tmp[4] + '_scaling_' + str( i+1 ) + '.csv' )

            elif args.aug_mode==2:
                augmentated_data = augmentation_x_translation( data )

                path = tmp[0] + '/' + tmp[1] + '/translation'
                check_and_mkdir( path )
                path = path + '/' + tmp[3]
                check_and_mkdir( path )

                writing_file( augmentated_data, item, path + '/' + tmp[4] + '_tranlation_' + str( i+1 ) + '.csv' )

            elif args.aug_mode==3:
                augmentated_data = augmentation_y_rotation( data )

                path = tmp[0] + '/' + tmp[1] + '/rotation'
                check_and_mkdir( path )
                path = path + '/' + tmp[3]
                check_and_mkdir( path )

                writing_file( augmentated_data, item, path + '/' + tmp[4] + '_rotation_' + str( i+1 ) + '.csv' )

            elif args.aug_mode==4:
                augmentated_data = augmentation_y_translation( data )

                path = tmp[0] + '/' + tmp[1] + '/translation_y'
                check_and_mkdir( path )
                path = path + '/' + tmp[3]
                check_and_mkdir( path )

                writing_file( augmentated_data, item, path + '/' + tmp[4] + '_tranlation_y_' + str( i+1 ) + '.csv' )



if __name__ == '__main__':
    main()