#!/bin/bash
a=0
until [ ! $a -lt 4000 ]
do
    echo $a
    python Show_Epoch.py -epoch $a -SMILES results/smiles/epoch_$a.smi
    #python Show_Epoch.py -epoch $a -image results/image/epoch_$a.png
    a=`expr $a + 1`
done