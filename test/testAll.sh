#!/bin/bash

INFILE="./inp.txt"
OUTPUT="n"

python ./pyfft.py ${INFILE} ${OUTPUT}
../cpp/bin/dft_seq.out ${INFILE} ${OUTPUT}
../cpp/bin/fft_rec_seq.out ${INFILE} ${OUTPUT}
../cpp/bin/fft_iter_seq.out ${INFILE} ${OUTPUT}





