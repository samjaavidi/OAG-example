#!/bin/bash
#SBATCH --job-name=multi_model_run
#SBATCH --array=0-13%1       # 14 models total, 1 at a time
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:2        
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=logs/model_%A_%a.out
#SBATCH --error=logs/model_%A_%a.err
#SBATCH --time=3-00:00:00

# models list
MODELS=('phi4:14b'  'gpt-oss:20b'  'gpt-oss:120b'  'nemotron-3-super:120b' 'gemma3:27b' 'gemma3:12b' 'gemma3:4b' 'gemma3:1b' 'deepseek-r1:14b'   'deepseek-r1:70b'  'llama3.1:8b' 'llama3.1:70b' 'mistral-small3.2:24b'  'mistral-medium-3.5:128b')

# following lines make sure that correct env is active
CURRENT_MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}


CONDA_PATH="/apps/miniforge/miniforge3/etc/profile.d/conda.sh"

if [ -f "$CONDA_PATH" ]; then
    source "$CONDA_PATH"
else
    echo "ERROR: Conda profile not found at $CONDA_PATH"
    exit 1
fi
##########
conda activate SJ
export PYTHONNOUSERSITE=1
# 5. Verification: Print which python is being used to your logs
which python
python --version
##########  end of env setup
module load ollama
# Start Ollama on a unique, non-default port per job to avoid colliding

export OLLAMA_HOST=127.0.0.1:$((12000 + SLURM_ARRAY_TASK_ID))
##################################################################
export OLLAMA_SCHED_SPREAD=true

export OLLAMA_NUM_PARALLEL=1
export OLLAMA_BATCH_SIZE=128

##################################################################
ollama serve &
until curl -s http://$OLLAMA_HOST/api/tags > /dev/null; do
    sleep 5
done

# Make sure the model is actually available on this Ollama instance
ollama pull "$CURRENT_MODEL"

# Precompute the patient-similarity matrix once it , The output then be used in code later
python /home/***/LLM/script/shots/compute_patient_similarity.py

# Run the Python inference
echo $CURRENT_MODEL
python /home/***/LLM/script/shots/predict_sz_outcome_similar_shots.py "$CURRENT_MODEL"

kill %1
