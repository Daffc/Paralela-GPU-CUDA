#!/bin/bash

export PATH=$PATH:/usr/local/cuda/bin &&\
#cd ./rainhas_serial && ./testes.sh && cd .. &&\
#cd ./rainhas_GPU && ./testes.sh && cd .. &&\
cd ./rainhas_GPU_2 && ./testes.sh && cd .. &&\
cd ./rainhas_GPU_3 && ./testes.sh && cd ..
