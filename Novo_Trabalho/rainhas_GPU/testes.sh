#!/bin/bash

echo "" > saidas_2
for i in {1..17}
do
    echo "======================= Rodada: "$i" ======================= "  >> saidas_2
    ../rainhas_iterativo $i >> saidas_2 
    ./rainhas_GPU $i >> saidas_2  
done