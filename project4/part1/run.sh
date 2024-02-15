#!/bin/bash

# Compile
javac KMeans.java

# Make tests dir
mkdir -p tests

# Run Tests for different K
# K=(2 5 10 15 20 50 100)
K=(2 5 10 15 20)
P=(1 2 3)
for i in ${K[@]}
do
  for p in ${P[@]}
  do
    echo "Running Koala $i $p"
    java KMeans Koala.jpg $i tests/Koala-$i-$p.jpg
    echo "Runing Penguin $i $p"
    java KMeans Penguins.jpg $i tests/Penguins-$i-$p.jpg
  done
done
