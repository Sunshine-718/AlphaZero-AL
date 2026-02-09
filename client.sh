#! /usr/bin/env bash

NUM=$1
shift
for i in $(seq 1 "$NUM"); do
    python3 client.py "$@" & 
done
wait