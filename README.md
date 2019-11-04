## Skelton viewer
In the case of you want to display one skelton data (a time)
$ python skelton_viewer.py --data_name [show file path] --show_id 100 --show_mode 0

In the case of you want to display one skelton data (all time)
$ python skelton_viewer.py --data_name [show file path] --show_id 100 --show_mode 1

In the case of you want to display two skelton data (a time)
$ python skelton_viewer.py --data_name [show file path] --show_id 100 --show_mode 2 --second_name [show file path]

## Data Augmentation
$ python data_augmentation.py --aug_mode 0 --target_dataset original

## Data Normalization
$ python data_normalization.py --target_dataset all

## Data Split
$ python data_normalization.py --target_dataset normalize --split_size 50


## Dataset
dataset - original              : raw data
        - augmentation          : augmented data
                  - noise       : add gaussian noise (N(0,5)) to each node position (x,y,z)
                  - scaling     : z-value scaling (range[-100,100])
                  - translation : x-value translation (range[-100,100])
                  - rotation    : rotation around y axis (range[-45,45])
        - all                   : raw data + augmented data
        - normalize             : data normalized to all dataset
        - split_50              : data split into 50 time series
        - split_100             : data split into 100 time series


## Docker
$ sudo docker build -t omoro_docker .

$ sudo docker run --runtime=nvidia -v /home/{path to omoro_featival folder}/omoro_festival:/home/contuser -it omoro_docker

## LSTM
### Train
$ python3 train.py --save_model_name model_cpu.ckpt --epoch 50

### Inference
$ python3 inference.py --model_name model_cpu.ckpt --test_name test.csv 
