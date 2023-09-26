#!/bin/sh

sbatch test.slurm

sleep 5s

cat outputs/test.$1.out
