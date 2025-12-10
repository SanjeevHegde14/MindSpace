# MindSpace

MindSpace is a web application designed to analyze textual data for mental health risk indicators using Natural Language Processing (NLP). The app allows users to input text or analyze conversation history to detect potential mental health patterns based on a trained Naive Bayes classifier.

## Overview

The project consists of a Flask web interface and a data science pipeline. The model was trained on dataset aggregations (including Reddit data) to classify text into specific mental health categories.

**Key Features:**
* **Text Analysis Engine:** Real-time sentiment and risk assessment using a serialized Naive Bayes model (`model_nb.pkl`).
* **User Dashboard:** specialized views for inputting text, viewing results, and tracking history.
* **Data Extraction Scripts:** Jupyter notebooks included for scraping and cleaning data from Reddit and parsing WhatsApp chat logs.
* **Risk Information:** Educational resources and risk categorization logic.

## Tech Stack

* **Backend:** Python 3, Flask
* **Frontend:** HTML5, CSS (Jinja2 Templates)
* **Database:** SQLite (`mindspace.db`)
* **Machine Learning:** Scikit-learn, Pandas, NLTK
* **Data Sources:** Reddit API, local chat exports

## Project Structure

```bash
├── app.py                  # Main Flask application entry point
├── instance/               # SQLite database storage
├── content/                # Datasets and serialized models (.pkl)
├── templates/              # HTML files (Login, Results, Dashboard)
├── anaconda_projects/      # DB management files
├── *.ipynb                 # Notebooks for model training & data extraction
└── ...
```
Some screenshots:
<img width="942" height="221" alt="image" src="https://github.com/user-attachments/assets/94a339c1-c7fe-4c08-8988-10bebd52ff70" />
<img width="641" height="363" alt="image" src="https://github.com/user-attachments/assets/00ae64c4-5609-4a60-b33e-e9525a32a928" />
<img width="942" height="298" alt="image" src="https://github.com/user-attachments/assets/4c805b4b-f555-45c1-8f7c-b0f1408162e5" />


