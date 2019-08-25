#!/bin/bash

#python ./src/find_best_layer.py --normalize="none" --model="mobilenet" --nr_images=810;
#python ./src/find_best_layer.py --normalize="none" --model="mobilenet" --nr_images=1620;
#python ./src/find_best_layer.py --normalize="none" --model="mobilenet" --nr_images=2430;
#python ./src/find_best_layer.py --normalize="none" --model="mobilenet" --nr_images=3240;

#python ./src/find_best_layer.py --normalize="minmax" --model="resnet" --nr_images=810;
#python ./src/find_best_layer.py --normalize="none" --model="mobilenet_v2" --nr_images=810;
#python ./src/find_best_layer.py --normalize="none" --model="mobilenet_v2" --nr_images=1620;
#python ./src/find_best_layer.py --normalize="none" --model="mobilenet_v2" --nr_images=2430;
#python ./src/find_best_layer.py --normalize="none" --model="mobilenet_v2" --nr_images=3240;

python ./src/find_best_layer.py --normalize="minmax" --model="resnet" --nr_images=1620;
python ./src/find_best_layer.py --normalize="standard" --model="resnet" --nr_images=1620;
python ./src/find_best_layer.py --normalize="minmax" --model="resnet" --nr_images=2430;
python ./src/find_best_layer.py --normalize="standard" --model="resnet" --nr_images=2430;