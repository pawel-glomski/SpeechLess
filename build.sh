#!/bin/bash
rm -r ./bin
mkdir ./bin && cd bin && cmake .. && cmake --build .