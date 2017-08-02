#!/bin/bash

INFILE="./inp.txt"
STATFILE="./stats.txt"
OUTPUT="y"

#echo "" > ${STATFILE}

for n in {14..15}
do

echo "Generating new input file for n = $n"
python randomInput.py $n > ${INFILE}

echo "  Running numpy fft"
python ./pyfft.py ${INFILE} ${OUTPUT} > numpy_result
head -n 2 numpy_result
numpy_time=`head -n 2 numpy_result | grep time | cut -f 2 -d ':' | sed -e 's/\s\+//g'`


if [ $n -lt 8 ] 
then
	echo "Running sequential dft"
	../cpp/bin/dft_seq.out ${INFILE} ${OUTPUT} > dftResult
	head -n 2 dftResult
	dft_time=`head -n 2 dftResult | grep time | cut -f 2 -d ':' | sed -e 's/\s\+//g'`
else
	dft_time=0
fi

#echo "  Running sequential recursive fft"
#../cpp/bin/fft_rec_seq.out ${INFILE} ${OUTPUT} > fft_rec_result
#head -n 2 fft_rec_result
#fft_rec_time=`head -n 2 fft_rec_result | grep time | cut -f 2 -d ':' | sed -e 's/\s\+//g'`

echo "  Running sequential iterative fft"
../cpp/bin/fft_iter_seq.out ${INFILE} ${OUTPUT} > fft_iter_result
head -n 2 fft_iter_result
fft_iter_time=`head -n 2 fft_iter_result | grep time | cut -f 2 -d ':' | sed -e 's/\s\+//g'`

#echo "  Running cilk recursive fft"
#../cilk/bin/fft_rec_cilk.out ${INFILE} ${OUTPUT} > fft_rec_cilk_result
#head -n 2 fft_rec_cilk_result

#echo "  Running cilk iterative fft"
#../cilk/bin/fft_iter_cilk.out ${INFILE} ${OUTPUT} > fft_iter_cilk_result
#head -n 2 fft_iter_cilk_result

echo "  Running cuda fft"
../cuda/bin/fft_mycuda.out ${INFILE} ${OUTPUT} > fft_cuda_result
head -n 2 fft_cuda_result
fft_cuda_time=`head -n 2 fft_cuda_result | grep time | cut -f2 -d':' | sed -e 's/\s\+//g' | cut -d'/' -f1`
fft_cuda_inner_time=`head -n 2 fft_cuda_result | grep time | cut -f2 -d':' | sed -e 's/\s\+//g' | cut -d'/' -f2`

#echo "  Running cufft fft"
#../cuda/bin/fft_cufft.out ${INFILE} ${OUTPUT} > fft_cufft_result
#head -n 2 fft_cufft_result
#fft_cufft_time=`head -n 2 fft_cufft_result | grep time | cut -f2 -d':' | sed -e 's/\s\+//g' | cut -d'/' -f1`
#fft_cufft_inner_time=`head -n 2 fft_cufft_result | grep time | cut -f2 -d':' | sed -e 's/\s\+//g' | cut -d'/' -f2`


# compare cpp to numpy
#echo "  Comparing cpp rec to numpy"
#python compareResults.py numpy_result fft_rec_result

# compare cilk recur to numpy
#echo "  Comparing cilk recur to numpy"
#python compareResults.py numpy_result fft_rec_cilk_result

# compare cilk to numpy
#echo "  Comparing cilk iter to numpy"
#python compareResults.py numpy_result fft_iter_cilk_result

# compare cuda to numpy
echo "  Comparing cuda to numpy"
python compareResults.py numpy_result fft_cuda_result


# write times to simple csv table
echo "$n, $numpy_time, $dft_time, $fft_rec_time, $fft_iter_time, $fft_cuda_time, $fft_cuda_inner_time, $fft_cufft_time, $fft_cufft_inner_time" >> ${STATFILE}

echo ""
echo ""
done

