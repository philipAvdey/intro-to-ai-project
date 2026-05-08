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