#!/bin/bash
#SBATCH --account=def-revins
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1000mb
#SBATCH --output=%x-%j.out

echo "Current work dir: `pwd`"
echo "Starting run at: `date`"

echo "Job ID: $SLURM_JOB_ID"

echo "prog started at: `date`"
mpiexec python cluster.py
echo "prog ended at: `date`"
