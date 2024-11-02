#!/bin/bash

echo "building!"

g++ -o main src/main.cpp -I/usr/include/opencv4 -L/usr/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc
