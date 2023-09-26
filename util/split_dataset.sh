#!/bin/bash

# Check if a file name is provided
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

input_file="$1"

# Calculate the number of lines in the file
total_lines=$(wc -l < "$input_file")

# Calculate 80%, 10% and 10% of the total lines
part1_lines=$(( total_lines * 8 / 10 ))
part2_lines=$(( total_lines / 10 ))
# part3_lines is also 10%, but for precision, we'll derive it from total - part1 - part2
part3_lines=$(( total_lines - part1_lines - part2_lines ))

# Shuffle the file and redirect shuffled content to a temporary file
shuf "$input_file" > "${input_file}.shuffled"

# Split the shuffled file into three files based on calculated line numbers
head -n "$part1_lines" "${input_file}.shuffled" > "${input_file}.part1"
tail -n $((part2_lines + part3_lines)) "${input_file}.shuffled" | head -n "$part2_lines" > "${input_file}.part2"
tail -n "$part3_lines" "${input_file}.shuffled" > "${input_file}.part3"

# Optional: remove the temporary shuffled file
rm "${input_file}.shuffled"
