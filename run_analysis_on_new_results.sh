#!/usr/bin/env bash

cp original_results/Analysis* .

find . -maxdepth 1 -name "Analysis*.py" | while read line; do
    echo "$line" >> result_analysis.txt
    python3 "$line" >> result_analysis.txt
done

echo "Please check file result_analysis.txt for the textual results and the pdf files for the figures."
