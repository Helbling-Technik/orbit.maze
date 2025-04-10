#!/bin/bash
# run it with ". activateConda.sh"

# Source Miniconda
if [ -f "$HOME/miniconda3/bin/activate" ]; then
    source "$HOME/miniconda3/bin/activate"
else
    echo "Miniconda not found at ~/miniconda3. Please check your installation."
    exit 1
fi

# Activate the conda environment
conda activate isaaclab

# Check if activation was successful
if [ $? -eq 0 ]; then
    echo "Successfully activated Conda environment: isaaclab"
else
    echo "Failed to activate Conda environment: isaaclab. Check if it exists."
    exit 1
fi