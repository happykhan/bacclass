# 🧬 BacClass: Bacterial Biosample Classifier

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![NCBI](https://img.shields.io/badge/Data-NCBI%20API-orange.svg)](https://ncbi.nlm.nih.gov)

**BacClass** is an AI-powered classification tool that automatically categorizes bacterial biosample metadata from NCBI and ENA databases into different sources. 


## 📋 Classification Categories
BacClass classifies Salmonella biosamples into the following 11 categories, based on isolation source and context:

| #  | Category            | Description                                                         | Examples                                             |
|----|---------------------|---------------------------------------------------------------------|------------------------------------------------------|
| 1  | **Clinical**        | Human/animal patients, clinical specimens, hospitals, medical labs  | Blood cultures, wound swabs, hospital isolates       |
| 2  | **Environmental**   | Natural environments, wildlife, non-food, non-clinical settings     | Water, soil, sewage, wildlife samples                |
| 3  | **Food**            | Food products, food processing/production environments              | Meat, dairy, produce, food facility swabs            |
| 4  | **Companion Animal**| Pets or companion animals, veterinary clinical samples              | Dog/cat feces, veterinary clinic samples             |
| 5  | **Aquatic animal**  | Aquatic animals (fish, shellfish, etc.)                            | Fish tissue, shrimp, aquaculture samples             |
| 6  | **Animal feed**     | Animal feed or feed production environments                        | Feed pellets, feed mill swabs                        |
| 7  | **Laboratory**      | Laboratory or research environments, experimental setups            | Lab strains, experimental cultures                   |
| 8  | **Livestock**       | Livestock or farm animals (excluding poultry)                       | Cattle, swine, sheep, goat samples                   |
| 9  | **ND**              | Not determined or not applicable                                   | Unspecified or ambiguous sources                     |
| 10 | **Poultry**         | Poultry or poultry production environments                         | Chicken, turkey, egg, poultry farm samples           |
| 11 | **Wild animal**     | Wild animals or wildlife habitats                                  | Deer, rodents, wild bird samples                     |

Each biosample is assigned to exactly one category based on all available metadata (isolation source, host, sample type, location, etc.).


## 🛠️ Installation

### Prerequisites
- Python 3.11 or higher
- OpenAI API key
- Google Bigquery

### Quick Install

```bash
# Clone the repository
git clone https://github.com/happykhan/bacclass.git
cd bacclass

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your credentials
```

### Environment Configuration

Create a `.env` file with your API credentials:

```env
# Required: OpenAI API key
OPENAI_API_KEY=sk-your-openai-api-key-here

```

---

## 🎯 Quick Start

Retrieve a dataset of Salmonella biosamples and classify them:

```bash
python bacclass.py prepare-dataset
python bacclass.py run-classifier --csv-path salmonella_biosamples.csv --output classification_results.csv
```
## 📊 Performance Benchmarks

### Overall Performance Metrics

| Metric                | Value    |
|-----------------------|----------|
| **Total Samples**     | 1000     |
| **Number of Classes** | 10       |
| **Accuracy**          | 0.9710   |
| **Macro F1 Score**    | 0.9536   |
| **Weighted F1 Score** | 0.9720   |
| **Macro Precision**   | 0.9554   |
| **Macro Recall**      | 0.9636   |
| **Cohen's Kappa**     | 0.9418   |
| **Cost/1000 samples** | ~$1.50   |

### Per-Class Performance

| Class            | Precision | Recall  | F1-Score | Support |
|------------------|-----------|---------|----------|---------|
| Animal Feed      | 1.0000    | 1.0000  | 1.0000   | 5       |
| Aquatic Animal   | 1.0000    | 1.0000  | 1.0000   | 1       |
| Clinical         | 0.9928    | 0.9986  | 0.9957   | 690     |
| Environmental    | 1.0000    | 1.0000  | 1.0000   | 38      |
| Food             | 0.9917    | 0.8623  | 0.9225   | 138     |
| Laboratory       | 1.0000    | 1.0000  | 1.0000   | 9       |
| Livestock        | 0.9268    | 1.0000  | 0.9620   | 38      |
| ND               | 1.0000    | 0.7750  | 0.8732   | 40      |
| Poultry          | 0.6429    | 1.0000  | 0.7826   | 36      |
| Wild Animal      | 1.0000    | 1.0000  | 1.0000   | 5       |

### Confidence Analysis

- **High Confidence:** 964 samples (96.4%)
- **Medium Confidence:** 9 samples (0.9%)
- **Low Confidence:** 27 samples (2.7%)

**Accuracy by Confidence Level:**
- High: 0.9741 (n=964)
- Medium: 0.5556 (n=9)
- Low: 1.0000 (n=27)

### Most Common Misclassifications

- Food → Poultry: 19 cases
- ND → Clinical: 5 cases
- ND → Livestock: 3 cases
- Clinical → Poultry: 1 case
- ND → Food: 1 case

## 📈 Evaluation Plots

The following plots are available in the `evaluation_results/plots` directory:

- **Confusion Matrix:** Visualizes the classifier's performance across all categories.
- **Per-Class Precision/Recall/F1 Bar Charts:** Shows precision, recall, and F1-score for each class.
- **Confidence Distribution:** Displays the distribution of classification confidence levels.
- **Misclassification Heatmap:** Highlights the most frequent misclassifications between categories.

You can view these plots by opening the corresponding image files in `evaluation_results/plots`.