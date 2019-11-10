# パッケージのimport
import glob
import os.path as osp
import numpy as np
import time
from tqdm import tqdm
import argparse

from model import RNN
from utils import VATLoss

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
#import torchvision.transforms as transforms

import dataloader

class Train( object ):

    def __init__( self, model_name, target, sn=False ):
        # Device configuration
        self.device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )

        # Hyper-parameters
        self.__sequence_length = 50
        self.__input_size = 78
        self.__hidden_size = 256
        self.__num_layers = 3
        self.__num_classes = 7
        self.__batch_size = 3 #256

        self.model = RNN( self.__input_size, self.__hidden_size, self.__num_layers, self.__num_classes, sn ).to( self.device )
        self.param = torch.load( model_name )
        self.model.load_state_dict( self.param )

        self.data_load( target )


    def data_load( self, target ):
        # load data
        print( "Test target : {}".format( target ) )
        val_list = dataloader.make_datapath_list( phase=target )
        val_dataset = dataloader.LoadDataset( file_list=val_list, phase=target )
        self.val_loader = torch.utils.data.DataLoader( val_dataset, batch_size=self.__batch_size, shuffle=True )


    def test( self ):
        # Test the model
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.val_loader:
                images = images.reshape( -1, self.__sequence_length, self.__input_size ).to( self.device )
                labels = labels.to( self.device )
                outputs = self.model( images, self.device )
                _, predicted = torch.max( outputs.data, 1 )
                total += labels.size(0)
                correct += ( predicted == labels ).sum().item()
                #print(f'predicted={predicted}')
                #print(f'labels={labels}')

            print('Test Accuracy of the model on the {} test data: {} %'.format(total, 100 * correct / total))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument( "--model_name", type=str, help="Test model ckpt name (default: model.ckpt)", default='model.ckpt' )
    parser.add_argument( "--target_dataset", type=str, help="Test dataset (default: val)", default='val' )
    parser.add_argument( "--sn", type=bool, help="Use SN (default: False)", default=False )
    args = parser.parse_args()

    lstm_train = Train( args.model_name, args.target_dataset, args.sn )
    lstm_train.test()


if __name__ == '__main__':
    main()