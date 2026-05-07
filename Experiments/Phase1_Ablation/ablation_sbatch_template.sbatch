#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################
#SBATCH --partition=jelsana		        ### specify partition name where to run a job. change only if you have a matching qos!! main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time=03-00:00:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --output=%x_%J.out			### output log for running job - %x for job name, %J for job number
#SBATCH --gpus=rtx_6000:1
#SBATCH --tasks=6


# Note: the following 4 lines are commented out
#SBATCH --mail-user=ahmedmas@post.bgu.ac.il	### user's email for sending job status messages
#SBATCH --mail-type=ALL				### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE

################  Following lines will be executed by a compute node    #######################

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n"
echo -e "ENVIRONMENT:\t\t" $env
echo -e "PROJECT:\t\t" $model_dir "\n\n"
### Start your code below ####
cd ~/BGU-Lab/$model_dir
module load anaconda ### load anaconda module
eval "$(conda shell.bash hook)" ### initialize conda for bash
conda activate $env ### activate a conda environment, replace my_env with your conda environment	
./run_train.sh $JOB_ID
