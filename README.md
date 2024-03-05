# Text-Comparison-and-TF-IDF-Visualization

## Overview
This repository contains a Python script for comparing two paragraphs and visualizing their TF-IDF (Term Frequency-Inverse Document Frequency) scores. The script uses the scikit-learn library for TF-IDF vectorization and cosine similarity calculation, along with NLTK for text preprocessing.

## Features
- **Paragraph Comparison:** Enter two paragraphs and the script will compute their cosine similarity, indicating the degree of similarity between their objectives.

- **TF-IDF Visualization:** Visualize the TF-IDF scores for the top 10 terms in each paragraph, providing insights into the importance of terms in the context of the given paragraphs.

## Usage
1. Run the script
2. Enter the two paragraphs as prompted.
3. The script will output the cosine similarity between the paragraphs and display bar graphs for the TF-IDF scores of the top terms in each paragraph.

## Dependencies
- Python 3.x
- Matplotlib
- scikit-learn
- NLTK

Install the required dependencies using:
```bash
pip install matplotlib scikit-learn nltk
