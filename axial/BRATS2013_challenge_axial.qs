#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -j y
#$ -pe orte 2
#$ -q NLPR06
#$ -l h=g0602

matlab -nojvm -nodisplay -nodesktop -nosplash < BRATS2013_challenge_segment_axial.m
