#!/usr/bin/env python

import subprocess
import itertools as it
import time

N_JOBS = 4
N_EPOCHS = 2001
N_GPU = 2

if __name__ == "__main__":
    lr_disc = (1e-3, 1e-4, 1e-5)
    lr_gen = (1e-3, 1e-4, 1e-5)
    bsize = (64,128,256)

    npoints = len(lr_disc)*len(lr_gen)*len(bsize)

    gpu_queues = [[] for _ in range(N_GPU)]
    for ipoint,(lrd,lrg,b) in enumerate(it.product(lr_disc, lr_gen, bsize)):
        while all([len(gpu_jobs) >= N_JOBS for gpu_jobs in gpu_queues]):
            time.sleep(1)
            for gpu_jobs in gpu_queues:
                for j in gpu_jobs:
                    if j.poll() is not None:
                        gpu_jobs.remove(j)
                        break

        for gpuid,gpu_jobs in enumerate(gpu_queues):
            if len(gpu_jobs) < N_JOBS: break

        j = subprocess.Popen(map(str, ['python', 'rnngan-train.py',
            '--lr-gen', lrg,
            '--lr-disc', lrd,
            '--batch-size', b,
            '--epochs', N_EPOCHS,
            '--rotate',
            '--gpu', gpuid,
            'rg-search'
            ]))
        print("SUBMITTED JOB %d/%d"%(ipoint, npoints))
        gpu_queues[gpuid].append(j)
