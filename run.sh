#!/bin/bash

while true
do
    echo "开始运行 Python 脚本"
    python code/ml/step2_data_cleaning.py data/samples/econ-neg/data_sample.csv data/samples/econ-neg

    # 如果上面的python脚本因为任何原因退出，
    # 下面这行会被执行，然后再次进入循环。
    echo "Python 脚本退出，等待 5 秒后重启"
    sleep 5
done
