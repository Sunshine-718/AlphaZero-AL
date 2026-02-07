#! /usr/bin/env bash
NUM=$1            # 第一个参数：启动数量
shift             # 把参数列表往前移，剩下的都是 client.py 的参数

for i in $(seq 1 "$NUM"); do
    python3 client.py "$@" &
done
wait