# Sentiment Analysis with VADER and RoBERTa

This repository contains a single Jupyter notebook, `sentimentanalysis.ipynb`, that performs sentiment analysis on Amazon product reviews using two approaches: a lexicon‑based model (VADER) and a transformer‑based model (RoBERTa). It includes basic exploratory data analysis, runs both models on the same set of reviews, and compares their sentiment predictions with the original star ratings.

## Features

- Uses VADER for fast, rule‑based sentiment scoring.
- Uses a pretrained RoBERTa model from Hugging Face for deep learning sentiment classification.
- Works on a subset of the Amazon Fine Food Reviews dataset.
- Provides simple visualizations and comparisons between model scores and review ratings.

## Files

- `sentimentanalysis.ipynb` – main notebook with all analysis.
- `README.md` – this project description.

## Setup

1. Create and activate a Python environment.
2. Install required libraries (for example: `pandas`, `numpy`, `matplotlib`, `seaborn`, `nltk`, `transformers`, `scipy`, `torch`, `jupyter`).
3. Ensure the Amazon reviews CSV is available and update the file path in the notebook if needed.

## Usage

1. Launch Jupyter Notebook.
2. Open `sentimentanalysis.ipynb`.
3. Run the cells from top to bottom to load the data, compute VADER and RoBERTa scores, and view the plots and comparisons.

## Conclusion
![VADER vs RoBERTa pairplot](vader_roberta_pairplot.png)
- Both VADER and RoBERTa scores generally align with the Amazon star ratings: high‑star reviews cluster with low negative and high positive scores, while low‑star reviews show the opposite pattern.  
- RoBERTa produces more sharply separated negative/neutral/positive regions, suggesting it is more confident and discriminative than VADER, whose scores spread more widely and overlap between classes.  
- Overall, both methods capture sentiment direction reasonably well, but RoBERTa appears better suited when stronger separation between sentiment classes is required.
