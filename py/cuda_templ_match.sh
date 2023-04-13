module load conda/2022-09-08
conda activate numba
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.7.1
export NUMBA_CUDA_DRIVER=/soft/compilers/cudatoolkit/cuda-11.7.1/lib64/libcudart.so
echo "CUDA_HOME: ${CUDA_HOME}"
echo "NUMBA_CUDA_DRIVER: ${NUMBA_CUDA_DRIVER}"

echo "CUDA-device attributes"
/usr/bin/time -v python3 /eagle/BrainImagingML/apsage/n_template_match_gpu/py/query_gpu_attributes.py

echo "Numba-CUDA"
START=`date +"%s"`
/usr/bin/time -v python3 /eagle/BrainImagingML/apsage/n_template_match_gpu/py/cuda_numba_templ_match.py 
NOW=`date +"%s"`
echo $((NOW - START)) seconds
