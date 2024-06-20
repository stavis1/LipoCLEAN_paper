#!/bin/bash

#SBATCH -A birthright
#SBATCH -p burst
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --ntasks-per-node=1
#SBATCH -J SLT05_instrumentCrossTraining
#SBATCH --mem=150g
#SBATCH -t 20:00:00
#SBATCH -o ./SLT05_instrumentCrossTraining_%j-output.txt
#SBATCH -e ./SLT05_instrumentCrossTraining_%j-error.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stavis@vols.utk.edu

export PATH=/home/4vt/env_run/bin:$PATH
conda run -p env_run python SLT05_paperFigures_instrumentCrossTraining.py
