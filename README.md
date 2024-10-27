# News Classification with RoBERTa

This project leverages the Stanford OVAL CCNews dataset to classify news articles into predefined categories based on content, title, and publisher information. Using RoBERTa, a state-of-the-art language model, we fine-tune for a multi-class text classification task to accurately categorize articles.

## Project Overview

**Goal**: The objective is to familiarize with various Deep Learning subtasks, focusing on Natural Language Processing (NLP) and specifically multi-class text classification.

**Dataset**:  
- Source: [Stanford OVAL CCNews on Hugging Face](https://huggingface.co/datasets/stanford-oval/ccnews)
- Columns used:
  - `plain_text`: Main content of the article.
  - `categories`: Target column for classification.
  - `title`: Title for additional context.
  - `publisher`: Publisher information as auxiliary data.

**Model**:  
- Pre-trained RoBERTa model fine-tuned on the CCNews dataset for improved performance in text classification.

## Project Structure

1. **Data Preprocessing**:  
   - Preprocessing includes cleaning and formatting `plain_text`, `title`, and `categories`.
   - Categories are standardized using Spacy for normalization and filtering of low-frequency labels.

2. **Text Composition**:
   - Several strategies, such as combining `title` and `plain_text`, are tested to identify the best structure for input text.

3. **Model Training**:
   - A custom training loop with early stopping and checkpointing is implemented to optimize the model.
   - Extended logging captures training and validation loss, along with accuracy metrics per epoch.

4. **Evaluation**:
   - The model is evaluated on validation and test sets, reporting accuracy, precision, recall, and F1 scores.
   - Cross-validation ensures robustness of the trained model.

## Code Structure

- **`load_dataset`**: Loads the dataset into a Spark DataFrame for scalability.
- **`clean_categories`** and **`standardize_categories`**: Functions for preprocessing categories.
- **`preprocess_data`**: Main function for data cleaning, text composition, and duplicate removal.
- **`train_model`**: Fine-tunes RoBERTa, implementing early stopping with checkpointing.
- **`evaluate_model`**: Evaluates the model using confusion matrix and per-category metrics.

## Getting Started

### Requirements

- Python 3.x
- PyTorch, Transformers, Scikit-learn, Optuna, and additional libraries as specified in `requirements.txt`.

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/news-classification.git
   cd news-classification
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook to preprocess data and train the model.

### Usage

- **Preprocess Data**: Load and clean the CCNews dataset.
- **Train the Model**: Fine-tune RoBERTa with early stopping enabled.
- **Evaluate**: Check performance metrics and visualize results.
