# 青山工业赛题镜像制作模板

## 容器镜像制作

### 构建模型
- 将数据处理、模型预测相关代码放到 `components` 文件夹下
- 将所需依赖写入 `requirements.txt` 中
- 文件结构可参照算盘中青山赛题验证模板中的文件结构

### 编辑修改 Dockerfile
- 按照需求修改 `docker/Dockerfile` 文件，增加需要安装的库或修改 pypi 镜像地址。

### 执行构建镜像命令
- 修改 `build.sh` 中镜像名称，镜像 tag，然后执行
```bash
./build.sh
```

- (可选择执行)运行容器测试,挂载本地数据集到容器内/code/components/data文件夹下（确保算法代码能读取这个路径，不能更改，便于后台自动流程）
```bash
docker run -it -v (本地目录):/code/components/data (镜像名) /bin/bash    
```
## 验证方法
- 本地环境下进入docker容器内 `/code` 目录下执行, 请仔细阅读修改 `run.sh` 内容
```bash
./run.sh
```
- 进入算盘，通过青山赛题验证模板完成验证，验证模板中已存在供选手验证的测试集

## 注意事项
- 官方测试集的目录及数据结构与青山赛题验证模板中 `/code/components/data` 路径文件结构一致
- 为了保证每个选手的最终结果不出现偏差，请不要在 `/code/components/data` 目录结构下多添加子文件夹
