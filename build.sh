#!/bin/bash
rm -r ./bin
mkdir ./bin && cd bin && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build .
