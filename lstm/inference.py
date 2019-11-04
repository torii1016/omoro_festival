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

    def __init__( self, model_name ):

        # Device configuration
        self.device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )

        # Hyper-parameters
        self.__sequence_length = 50
        self.__input_size = 78
        self.__hidden_size = 256
        self.__num_layers = 3
        self.__num_classes = 7

        self.model = RNN( self.__input_size, self.__hidden_size, self.__num_layers, self.__num_classes ).to( self.device )
        self.param = torch.load( model_name )
        self.model.load_state_dict( self.param )

    def loading_file( self, name ):
        with open( name ) as f:
            data = []
            reader = csv.reader(f)
            for j, row in enumerate( reader ):
                if j!=0:
                    data += row
                else:
                    item = row
        return np.array( data, dtype='float32' ) # data.shape: list[50*26*3] 1 demension vector
    
    def compute( self, input_data ):

        input_data = torch.from_numpy( input_data )
        with torch.no_grad():
            data = input_data.reshape( -1, self.__sequence_length, self.__input_size ).to( self.device )
            outputs = self.model( data, self.device )
            _, predicted = torch.max( outputs.data, 1 )
        return predicted
        

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument( "--model_name", type=str, help="Loading model ckpt name (default: model.ckpt)", default='model.ckpt' )
    parser.add_argument( "--test_name", type=str, help="Loading test file name (default: test.csv)", default='test.csv' )
    args = parser.parse_args()

    inference = Inference( args.model_name )
    data = inference.loading_file( args.test_name )

    predicted = inference.compute( data )

    print(f'predicted={predicted}')


if __name__ == '__main__':
    main()