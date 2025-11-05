#!/bin/bash

count=$(find ../training_data/blip/ -maxdepth 1 -type f -name "*.txt" | wc -l)
echo "Blip dataset pairs: $count"
