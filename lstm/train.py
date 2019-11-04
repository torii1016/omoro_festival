# パッケージのimport
import glob
import os.path as osp
import numpy as np
import time
from tqdm import tqdm
import argparse

from model import RNN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
#import torchvision.transforms as transforms

import dataloader

class Train( object ):

    def __init__( self, epoch ):
        # Device configuration
        self.device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )

        # Hyper-parameters
        self.__sequence_length = 50
        self.__input_size = 78
        self.__hidden_size = 256
        self.__num_layers = 3
        self.__num_classes = 7
        self.__batch_size = 256
        self.__num_epochs = epoch
        self.__learning_rate = 0.00005

        self.model = RNN( self.__input_size, self.__hidden_size, self.__num_layers, self.__num_classes ).to( self.device )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam( self.model.parameters(), lr=self.__learning_rate )

        self.data_load()


    def data_load( self ):
        # load data
        train_list = dataloader.make_datapath_list( phase="train" )
        val_list = dataloader.make_datapath_list( phase="val" )

        train_dataset = dataloader.LoadDataset( file_list=train_list, phase='train' )
        val_dataset = dataloader.LoadDataset( file_list=val_list, phase='val' )

        self.train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=self.__batch_size, shuffle=True )
        self.val_loader = torch.utils.data.DataLoader( val_dataset, batch_size=self.__batch_size, shuffle=True )


    def train( self, save_name ):

        # Train the model
        total_step = len( self.train_loader )
        for epoch in range( self.__num_epochs ):
            print(f'epoch = {epoch}')
            for i, (images, labels) in enumerate( self.train_loader ):
                #print(f'images={images}')
                #print(f'labels={labels}')
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model( images, self.device )
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #if (i+1) % 100 == 0:
                if (i+1) % self.__num_epochs == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format( epoch+1, self.__num_epochs, i+1, total_step, loss.item()) )

        # Save the model checkpoint
        torch.save( self.model.state_dict(), save_name )

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

            print('Test Accuracy of the model on the 7000 test data: {} %'.format(100 * correct / total))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument( "--epoch", type=int, help="Training epoch (default: 50)", default=50 )
    parser.add_argument( "--save_model_name", type=str, help="Save model ckpt name (default: model.ckpt)", default='model.ckpt' )
    args = parser.parse_args()

    lstm_train = Train( args.epoch )
    lstm_train.train( args.save_model_name )
    lstm_train.test()


if __name__ == '__main__':
    main()