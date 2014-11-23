#!/bin/bash

OUT_FILE=games.csv
head -1 data/ANA.csv > $OUT_FILE
cat data/*.csv |grep -v Faceoff >> $OUT_FILE
