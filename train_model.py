"""Compatibility entry point for training the saved BOP model bundle.

Usage from the project folder:
    python train_model.py --data data/Training_BOP_Cleaned.csv

The training code and Streamlit app share the same deterministic feature engineering,
preprocessing pipeline, class weighting and fixed 0.10 classification threshold.
"""
from src.train import main

if __name__ == "__main__":
    main()
