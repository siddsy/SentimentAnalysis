Sentiment Analysis with VADER and RoBERTa
This repository contains a single Jupyter notebook that demonstrates sentiment analysis using both a classic lexicon‑based method (VADER) and a modern transformer‑based model (RoBERTa) on Amazon review data.

Features
End‑to‑end sentiment analysis in one notebook.

VADER (Valence Aware Dictionary and sEntiment Reasoner) for fast, rule‑based sentiment scoring.

RoBERTa (via Hugging Face Transformers) for deep‑learning‑based sentiment classification.

Exploratory data analysis (EDA) of ratings and review text.

Side‑by‑side comparison of VADER and RoBERTa outputs.

Project Structure
text
.
├── sentimentanalysis.ipynb   # Main analysis notebook
└── README.md                 # Project documentation (this file)
You can optionally add:

text
data/
  Reviews.csv                 # Amazon Fine Food Reviews subset (if stored locally)
requirements.txt              # Python dependencies
Data
The notebook uses a subset of the Amazon Fine Food Reviews dataset, which includes fields such as:

Score – 1–5 star rating

Summary – short review title

Text – full review content

The notebook reads the CSV from a path you can adapt to your environment (e.g. Kaggle input path or a local data/Reviews.csv).

Methods
VADER

Uses NLTK’s SentimentIntensityAnalyzer.

Produces four scores per review: neg, neu, pos, compound.

Merges scores back into the original DataFrame and visualizes how they relate to star ratings.

RoBERTa

Uses a pretrained RoBERTa sentiment model from Hugging Face (e.g. cardiffnlp/twitter-roberta-base-sentiment).

Tokenizes each review, runs it through the model, and extracts probabilities for negative, neutral, and positive sentiment.

Stores probabilities in new columns and compares them to VADER scores and star ratings.

Comparison

The notebook includes plots and tables to:

Inspect rating distributions.

Inspect the distribution of VADER compound scores.

Compare VADER and RoBERTa behaviour across different star levels.

Requirements
Create a requirements.txt similar to:

text
pandas
numpy
matplotlib
seaborn
nltk
transformers
tqdm
scipy
jupyter
Install with:

bash
pip install -r requirements.txt
Inside the notebook, ensure the VADER lexicon is available:

python
import nltk
nltk.download("vader_lexicon")
How to Run
Clone the repository

bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
Install dependencies

bash
pip install -r requirements.txt
Add the data

Option A (local): Place Reviews.csv under data/ and update the path in the notebook.

Option B (Kaggle): Attach the dataset as an input and keep the existing path.

Start Jupyter

bash
jupyter notebook
Open and run

Open sentimentanalysis.ipynb and run cells from top to bottom.

Possible Extensions
Swap in a different Hugging Face sentiment model.

Run on a larger subset or the full dataset and log results to CSV.

Add evaluation metrics (accuracy, F1) against the star labels.

Wrap the models into a small API or Streamlit/Gradio demo.

