import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import argparse

def loading_file( name ):

    data = np.empty( (0,26,3) , float )
    spilit_list = [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75]
    with open( name ) as f:
        reader = csv.reader(f)
        for j, row in enumerate( reader ):
            if j!=0:
                tmp = np.split(row,spilit_list)
                data = np.append( data, [tmp], axis=0 )

    return data

def loading_hierarchy( name ):

    hierarchy = np.empty( (0,2) , int )
    f = open( name )
    reader = f.readlines()
    f.close()
    for line in reader:
        tmp_ = line.split( '\n' )
        tmp = tmp_[0].split( ',' )
        hierarchy= np.append( hierarchy, [tmp], axis=0 )
        
    return hierarchy


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument( "--data_name", help="csv file name", default="default.csv" )
    parser.add_argument( "--second_name", help="csv file name", default="default.csv" )
    parser.add_argument( "--joint_h", help="joint hierarchy file name", default="../joint_hierarchy.txt" )
    parser.add_argument( "--show_id", help="show skelton data id", default=0, type=int )
    parser.add_argument( "--show_mode", help="show mode, single[0] or all[1], multi[2]", default=0, type=int )
    args = parser.parse_args()

    data = loading_file( args.data_name )
    hierarchy = loading_hierarchy( args.joint_h )

    data = np.array( data, dtype=float )

    print( args.show_mode )

    if args.show_mode==0 or args.show_mode==2:

        fig = plt.figure()
        ax = Axes3D(fig)

        for i in range(25):
            x = [float( data[int( args.show_id ),int( hierarchy[i,0] ),0] ),float( data[int( args.show_id ),int( hierarchy[i,1] ),0] )]
            y = [float( data[int( args.show_id ),int( hierarchy[i,0] ),1] ),float( data[int( args.show_id ),int( hierarchy[i,1] ),1] )]
            z = [float( data[int( args.show_id ),int( hierarchy[i,0] ),2] ),float( data[int( args.show_id ),int( hierarchy[i,1] ),2] )]
            ax.plot( x, y, z, marker='.', markersize=1, color='r', linewidth=2.0 )

        ax.plot( data[int( args.show_id ),:,0], data[int( args.show_id ),:,1], data[int( args.show_id ),:,2], marker='.', markersize=10, color='b', linestyle='None' )

        if args.show_mode==2:
            data_ = loading_file( args.second_name )
            data_ = np.array( data_, dtype=float )

            for i in range(25):
                x = [float( data_[args.show_id,int( hierarchy[i,0] ),0] ),float( data_[args.show_id,int( hierarchy[i,1] ),0] )]
                y = [float( data_[args.show_id,int( hierarchy[i,0] ),1] ),float( data_[args.show_id,int( hierarchy[i,1] ),1] )]
                z = [float( data_[args.show_id,int( hierarchy[i,0] ),2] ),float( data_[args.show_id,int( hierarchy[i,1] ),2] )]
                ax.plot( x, y, z, marker='.', markersize=1, color='g', linewidth=2.0 )

            ax.plot( data_[args.show_id,:,0], data_[args.show_id,:,1], data_[args.show_id,:,2], marker='.', markersize=10, color='g', linestyle='None' )

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    elif args.show_mode==1:

        for j in range( data.shape[0] ):

            fig = plt.figure()
            ax = Axes3D(fig)

            for i in range(25):
                x = [float( data[j,int( hierarchy[i,0] ),0] ),float( data[j,int( hierarchy[i,1] ),0] )]
                y = [float( data[j,int( hierarchy[i,0] ),1] ),float( data[j,int( hierarchy[i,1] ),1] )]
                z = [float( data[j,int( hierarchy[i,0] ),2] ),float( data[j,int( hierarchy[i,1] ),2] )]
                ax.plot( x, y, z, marker='.', markersize=1, color='r', linewidth=2.0 )

            ax.plot( data[j,:,0], data[j,:,1], data[j,:,2], marker='.', markersize=20, color='r', linestyle='None' )

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.show()
    

if __name__ == '__main__':
    main()