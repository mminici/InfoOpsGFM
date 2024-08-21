#!/bin/bash

# Change the current directory
cd ../src || { echo "Failed to change directory to ../src"; exit 1; }


# Check if the required parameters are passed
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <country> <device> <embed_type>"
    exit 1
fi

# Assign the parameters to variables
country="$1"
device="$2"
embed_type="$3"

# Define the output file using the country parameter
output_file="../script_output/normalGNN_${country}.txt"

# Clear the output file if it exists
> "$output_file"

# List of Python scripts to run with their arguments
scripts=(
    "python run_GNN.py --dataset $country --device $device --lr 1e-2 --early 20 --gnn sage --embed_type $embed_type"
    "python run_GNN.py --dataset $country --device $device --lr 1e-3 --early 20 --gnn sage --embed_type $embed_type"
    "python run_GNN.py --dataset $country --device $device --lr 1e-2 --early 20 --gnn gcn --embed_type $embed_type"
    "python run_GNN.py --dataset $country --device $device --lr 1e-3 --early 20 --gnn gcn --embed_type $embed_type"
    "python run_GNN.py --dataset $country --device $device --lr 1e-2 --early 25 --gnn sage --embed_type $embed_type"
    "python run_GNN.py --dataset $country --device $device --lr 1e-3 --early 25 --gnn sage --embed_type $embed_type"
    "python run_GNN.py --dataset $country --device $device --lr 1e-2 --early 25 --gnn gcn --embed_type $embed_type"
    "python run_GNN.py --dataset $country --device $device --lr 1e-3 --early 25 --gnn gcn --embed_type $embed_type"
    "python run_GNN.py --dataset $country --device $device --lr 1e-2 --early 30 --gnn sage --embed_type $embed_type"
    "python run_GNN.py --dataset $country --device $device --lr 1e-3 --early 30 --gnn sage --embed_type $embed_type"
    "python run_GNN.py --dataset $country --device $device --lr 1e-2 --early 30 --gnn gcn --embed_type $embed_type"
    "python run_GNN.py --dataset $country --device $device --lr 1e-3 --early 30 --gnn gcn --embed_type $embed_type"

)


# Iterate through the scripts
for script in "${scripts[@]}"; do
    # Construct the full command to be executed
    full_command="$script"

    # Print the command to be run
    echo "Running: $full_command"

    # Append the separator, script command, and --under value to the output file
    echo "=====" >> "$output_file"
    echo "$full_command" >> "$output_file"

    # Run the script with the --under value and append its output to the output file
    $full_command >> "$output_file" 2>&1

    # Append the END separator with the script name and --under value to the output file
    echo "END of $full_command" >> "$output_file"
done
