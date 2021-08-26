# constants.py

import os

# Project directory
PROJECT_DIR = os.path.split(os.getcwd())[0]

# Data directories
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
RAW_IMAGES_DIR = os.path.join(RAW_DATA_DIR, 'AML-Cytomorphology')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

# Reference directory
REFERENCES_DIR = os.path.join(PROJECT_DIR, 'references')
IMAGE_STATS = 'image_stats.csv'

# Reports directory
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
