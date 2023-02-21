#!/bin/bash

image=components-demo
tag=1.0.0
repo=${image}:${tag}

docker build -f docker/Dockerfile -t ${repo} .
