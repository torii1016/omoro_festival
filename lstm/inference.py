# パッケージのimport
import glob
import os.path as osp
import numpy as np
import time
from tqdm import tqdm
import csv
import argparse

from model import RNN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
#import torchvision.transforms as transforms


class Inference( object ):

    def __init__( self, model_name, sn=False ):

        # Device configuration
        self.device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )

        # Hyper-parameters
        self.__sequence_length = 50
        self.__input_size = 78
        self.__hidden_size = 256
        self.__num_layers = 3
        self.__num_classes = 7

        # min_max-parameters
        self.__x_min = -2259.0780484289285
        self.__x_max = 2548.842436486494
        self.__y_min = -1186.3449394557435
        self.__y_max = 939.5449823147761
        self.__z_min = 1000.04
        self.__z_max = 3323.48
        self.__v_min = np.array( [self.__x_min,self.__y_min,self.__z_min] )
        self.__v_max = np.array( [self.__x_max,self.__y_max,self.__z_max] )
        self.__max_min = self.__v_max-self.__v_min

        self.model = RNN( self.__input_size, self.__hidden_size, self.__num_layers, self.__num_classes, sn ).to( self.device )
        self.param = torch.load( model_name )
        self.model.load_state_dict( self.param )
        self.joint_heirarchy_length = self.loading_joint_hierarchy( "../joint_hierarchy.txt", "../joint_length.txt" )

    def loading_file( self, name ):

        data = np.empty( (0,26,3) , float )
        spilit_list = [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75]
        with open( name ) as f:
            reader = csv.reader(f)
            for j, row in enumerate( reader ):
                if j!=0 and j<=50:
                    tmp = np.split(row,spilit_list)
                    data = np.append( data, [tmp], axis=0 )
                else:
                    item = row

        return data

    def loading_joint_hierarchy( self, name_connection, name_length ):

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


    def size_normalization( self, data ):

        normalized_data = np.copy( data )

        for i, tmp_1 in enumerate( data ):
            for tmp_2 in self.joint_heirarchy_length:
                v = tmp_1[tmp_2[1]] - tmp_1[tmp_2[0]]
                v_ = v / np.linalg.norm(v)
                normalized_data[i][tmp_2[1]] = normalized_data[i][tmp_2[0]] + ( v_*tmp_2[2] )

        return normalized_data.reshape( -1 )

    def normalization( self, input_data ):

        mins = np.reshape( self.__v_min.tolist()*int( input_data.shape[0]/3 ), input_data.shape )
        output_data = ( input_data - mins )/( np.array( self.__max_min.tolist()*int( input_data.shape[0]/3 ) ) )
        return output_data.astype(np.float32)
    
    def compute( self, test_name ):

        data = self.loading_file( test_name )
        data = np.array( data, dtype=float )
        print( data.shape )

        normalized_data = self.size_normalization( data )
        print( normalized_data.shape )
        normalized_data = self.normalization( normalized_data )

        input_data = torch.from_numpy( normalized_data )
        with torch.no_grad():
            data = input_data.reshape( -1, self.__sequence_length, self.__input_size ).to( self.device )
            outputs = self.model( data, self.device )
            tmp = outputs.data*1000.0
            print( tmp.int()/1000.0 )
            _, predicted = torch.max( outputs.data, 1 )
        return predicted
        

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument( "--model_name", type=str, help="Loading model ckpt name (default: model.ckpt)", default='model.ckpt' )
    parser.add_argument( "--test_name", type=str, help="Loading test file name (default: test.csv)", default='test.csv' )
    parser.add_argument( "--sn", type=bool, help="Use SN (default: False)", default=False )
    args = parser.parse_args()

    inference = Inference( args.model_name, args.sn )
    predicted = inference.compute( args.test_name )

    print(f'predicted={predicted}')


if __name__ == '__main__':
    main()