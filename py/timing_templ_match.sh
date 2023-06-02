#qsub -A BrainImagingML -q debug -l select=1 -l walltime=00:60:00 -k doe -l filesystems=home:eagle -N timing_tmpl_match  timing_templ_match.sh
module load conda/2022-09-08
conda activate
echo "CUDA_HOME: ${CUDA_HOME}"
echo "NUMBA_CUDA_DRIVER: ${NUMBA_CUDA_DRIVER}"
cd /eagle/BrainImagingML/apsage/n_template_match_gpu/py/
echo "Numba"
START=`date +"%s"`
/usr/bin/time -v python3 /eagle/BrainImagingML/apsage/n_template_match_gpu/py/numba_templ_match.py 
NOW=`date +"%s"`
echo $((NOW - START)) seconds

echo "Scipy"
START=`date +"%s"`
/usr/bin/time -v python3 /eagle/BrainImagingML/apsage/n_template_match_gpu/py/scipy_templ_match.py 
NOW=`date +"%s"`
echo $((NOW - START)) seconds

echo "TF"
START=`date +"%s"`
/usr/bin/time -v python3 /eagle/BrainImagingML/apsage/n_template_match_gpu/py/tf_templ_match.py 
NOW=`date +"%s"`
echo $((NOW - START)) seconds

echo "cv-cpu"
START=`date +"%s"`
/usr/bin/time -v python3 /eagle/BrainImagingML/apsage/n_template_match_gpu/py/cv_templ_match.py 
NOW=`date +"%s"`
echo $((NOW - START)) seconds
