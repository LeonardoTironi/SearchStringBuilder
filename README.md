# Search String Builder

This project implements a term expansion system that creates enhanced search queries using WordNet synonyms and semantic similarity ranking with Sentence Transformers embeddings.

## Overview

The system takes input terms and expands them into semantically similar phrases to build search strings.

## Features

- **WordNet Integration**: Leverages NLTK's WordNet corpus to find synonyms for individual words
- **Semantic Similarity Ranking**: Uses SentenceTransformers to rank candidate phrases by semantic similarity to the original term using their embeddings
- **Phrase Combination Generation**: Creates all possible combinations of synonyms for multi-word terms
- **Search String Construction**: Builds boolean search strings with OR and AND operators

## Dependencies

The script requires the following Python packages:
- `nltk` - For WordNet synonym extraction
- `sentence-transformers` - For semantic similarity computation
- `itertools` - For generating combinations (built-in)

## Installation

1. Install the required packages globally or inside a virtual environment:
```bash
pip install nltk sentence-transformers
```
## Usage
```bash
python main.py
```

### Function Description

#### `get_wordnet_synonyms(term)`
- Extracts synonyms for a single word using WordNet
- Returns a list including the original term and its synonyms

#### `rank_candidates(base_term, candidates, top_k=3)`
- Ranks candidate phrases by semantic similarity to the base term
- Uses cosine similarity between sentence embeddings
- Returns the top-k most similar candidates

#### `expand_terms_with_combinations(term, top_k=3)`
- Expands a multi-word term by finding synonyms for each word
- Generates all possible combinations
- Returns the top-k most semantically similar combinations

#### `build_search_string(terms_list, top_k=3)`
- Main function that processes a list of terms
- Creates a search string with proper OR/AND operators
- Each term becomes an OR clause, multiple terms are joined with AND

## Model Information

The system uses the `all-MiniLM-L6-v2` model from SentenceTransformers, because it has the highest rank for "Performance Sentence Embeddings".