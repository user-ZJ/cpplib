#!/bin/bash
# coding=utf-8
echo "Current date: $(date)"


name=XXX

rm -rf statistic.txt


while :
do
    cpuRatio=`top -b -c -n 1 | grep $name | grep -v grep | head -n 1 | awk '{sub(/^\s+|\s+$/, "")}1' | awk '{print $9}'`
    cpuMem=`ps -aux | grep $name | grep -v grep | head -n 1 | awk '{print $6/1024}'`
    gpuRatio=`nvidia-smi | grep % | grep -v grep | head -n 1 | awk '{print $13}' | sed -e 's/%//g'`
    gpuMem=`nvidia-smi | grep $name | grep -v grep | head -n 1 | awk '{print $8}' | sed -e 's/MiB//g'`
    echo "$cpuRatio $cpuMem $gpuRatio $gpuMem" >> statistic.txt
    sleep 1
done


