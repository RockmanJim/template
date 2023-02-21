#!/bin/bash

# 运行算法, 生成test_predict.csv文件到/code目录下（确保该路径，不得修改）
python components/hellow_world.py 2>&1 | tee run.log
python components/test.py 2>&1 | tee run.log

# 该行不可删除，生成于/code目录下：结束文件，不包含内容，用于触发后台自动执行评分
touch finish.txt
