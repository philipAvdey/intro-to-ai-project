# Cinepic: Cosine Similarity vs Transformer
## CS 471 - Introduction to Artifical Intelligence Final Project

A content-based movie recommendation system that compares two approaches:
a cosine similarity baseline against a Transformer encoder trained with
self-supervised learning.

## Overview

The user inputs 3 favorite movies. Both models build a taste profile from
those inputs and return the top 10 most similar movies from a dataset of
4,375 films. Results are displayed side by side for comparison.

## Models

**Cosine Similarity (Baseline)**
Converts each movie into a feature vector and finds the nearest neighbors
using cosine similarity. Fast, interpretable, no training required.

**Transformer Encoder (Deep Learning)**
Organizes each movie's features into 5 tokens: genres, keywords,
popularity, vote average, and release year. The model then learns relationships
between them through self-attention. Trained self-supervised via
reconstruction loss. 

## Features Used

- Genres (one-hot encoded)
- Keywords (TF-IDF, 100 features)
- Popularity (normalized)
- Vote average (normalized)
- Release year (normalized)

## Installation

```bash
git clone https://github.com/philipAvdey/intro-to-ai-project.git
cd intro-to-ai-project
pip install pandas scikit-learn scipy torch openpyxl
```

## Usage

Run from the project root:

```bash
python src/app/app.py
```

Enter 3 movie titles when prompted. Both models will run and print
recommendations side by side.

## Limitations
- Director and studio are not features, limiting style-based recommendations.
- Transformer results vary slightly between runs due to random initialization.

## Dataset

Filtered movie dataset with 4,375 films including budget, genres, keywords,
popularity, revenue, vote average, cast, and director.
