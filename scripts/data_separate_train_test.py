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

def check_and_mkdir( path ):
    if not os.path.exists( path ):
        os.mkdir( path )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument( "--target_dataset", help="separation dataset name", default="original" )
    parser.add_argument( "--test_num", help="test data size", default="500", type=int )
    args = parser.parse_args()

    files = glob.glob( "../dataset/" + args.target_dataset + "/**/*.csv", recursive=True )

    test_data_num = args.test_num

    for file_ in files:

        name = file_.split( '.csv' )[0]
        tmp = name.split( '/' )
        print( tmp[4] )

        data, item = loading_file( name + ".csv" )
        data = np.array( data, dtype=float )

        path = tmp[0] + '/' + tmp[1] + '/split_' + str( split_num )
        check_and_mkdir( path )
        path = path + '/' + tmp[3]
        check_and_mkdir( path )

        num = 0
        for i in range( 0, data.shape[0], split_num ):
            split_data = np.zeros( (split_num, 26, 3) )
            if split_num<=data.shape[0]-i:
                split_data = split_data + data[i:i+split_num]
            else:
                split_data[:( data.shape[0]-i ),:,:] = split_data[:( data.shape[0]-i ),:,:] + data[i:data.shape[0],:,:]
            
            writing_file( split_data, item, path + '/' + tmp[4] + '_split_' + str(num+1)  + '.csv' )
            num = num + 1

if __name__ == '__main__':
    main()