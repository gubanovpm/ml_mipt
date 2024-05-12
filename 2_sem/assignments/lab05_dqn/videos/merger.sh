#!/bin/bash

echo "" > videolist.txt

VIDEOS=`ls | grep .mp4 | sort -n`
for video in $VIDEOS; do
  echo "file ./$video" >> videolist.txt
done

ffmpeg -f concat -safe 0 -i videolist.txt -c copy result.mp4
