#!/bin/bash
cat statistic.txt|awk '{sum+=$1} END {print "cpuRatio avg= ", sum/NR}'
cat statistic.txt|awk 'BEGIN {max = 0} {if ($1+0>max+0) max=$1 fi} END {print "cpuRatio Max=", max}'
cat statistic.txt|awk 'BEGIN {min = 1999999} {if ($1+0<min+0) min=$1 fi} END {print "cpuRatio Min=", min}'
cat statistic.txt|awk '{sum+=$2} END {print "cpuMem = ", sum/NR}'
cat statistic.txt|awk '{sum+=$3} END {print "gpuRatio = ", sum/NR}'
cat statistic.txt|awk 'BEGIN {max = 0} {if ($3+0>max+0) max=$3 fi} END {print "gpuRatio Max=", max}'
cat statistic.txt|awk 'BEGIN {min = 1999999} {if ($3+0<min+0) min=$3 fi} END {print "gpuRatio Min=", min}'
cat statistic.txt|awk '{sum+=$4} END {print "gpuMem = ", sum/NR}'
