#!/bin/bash

INFILE="./inp.txt"
OUTPUT="y"

for n in {13..21}
do

echo "Generating new input file for n = $n"
python randomInput.py $n > ${INFILE}

echo "  Running numpy fft"
python ./pyfft.py ${INFILE} ${OUTPUT} > numpy_result
head -n 2 numpy_result

#echo "Running sequential dft"
#../cpp/bin/dft_seq.out ${INFILE} ${OUTPUT} > dtfResult

echo "  Running sequential recursive fft"
../cpp/bin/fft_rec_seq.out ${INFILE} ${OUTPUT} > fft_rec_result
head -n 2 fft_rec_result

echo "  Running sequential iterative fft"
../cpp/bin/fft_iter_seq.out ${INFILE} ${OUTPUT} > fft_iter_result
head -n 2 fft_iter_result

echo "  Running cilk recursive fft"
../cilk/bin/fft_rec_cilk.out ${INFILE} ${OUTPUT} > fft_rec_cilk_result
head -n 2 fft_rec_cilk_result

echo "  Running cilk iterative fft"
../cilk/bin/fft_iter_cilk.out ${INFILE} ${OUTPUT} > fft_iter_cilk_result
head -n 2 fft_iter_cilk_result

#echo "  Running cuda fft"
#../cuda/bin/fft_cuda.out ${INFILE} ${OUTPUT} > fft_cuda_result
#head -n 2 fft_cuda_result

# compare cpp to numpy
echo "  Comparing cpp rec to numpy"
python compareResults.py numpy_result fft_rec_result

# compare cilk recur to numpy
echo "  Comparing cilk recur to numpy"
python compareResults.py numpy_result fft_rec_cilk_result

# compare cilk to numpy
echo "  Comparing cilk iter to numpy"
python compareResults.py numpy_result fft_iter_cilk_result

# compare cuda to numpy
#echo "  Comparing cuda to numpy"
#python compareResults.py numpy_result fft_cuda_result

echo ""
echo ""
done

