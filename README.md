# Data Analysis and Visualization Group Assignment

## Overview
This project analyzes barriers to mobile money adoption in South Africa and selected Sub-Saharan African countries using data from the World Bank Open Data platform. The study focuses on two datasets, WB_FINDEX_FIN14A (reason: mobile money agents are too far away) and WB_FINDEX_FIN14D (reason: don't have enough money to use a mobile money account), collected in 2024. The analysis includes data preparation, numerical analysis, visualizations, and database integration, with the goal of identifying trends and proposing interventions to enhance financial inclusion.

## Project Structure
- *README.md*: This file, providing an overview and instructions.
- *report.docx*: A 7-page project report.
- *WB_FINDEX_FIN14A/*: Directory for raw data files (if extracted from the script).
- *WB_FINDEX_FIN14D/*: Directory for raw data files (if extracted from the script).
- *FinancialBarriers/*: Directory for the generated visualizations and database files (e.g., findex.db).

## Requirements
### Software
- Python 3.8+
- LaTeX distribution (e.g., TeX Live or MikTeX) for report compilation
- SQLite for database operations

### Python Libraries
- pandas
- numpy
- matplotlib
- seaborn
- sqlite3
- streamlit

Install dependencies using:
bash
pip install pandas numpy matplotlib seaborn


## Installation
1. Clone the repository or download the files:
   bash
   git clone <repository-https://github.com/Nkosinathiz/Financial-Barriers-to-Mobile-Money>
   cd data-analysis-visualization-ndta631
   
2. Ensure all required Python libraries are installed as listed above.
3. Install a LaTeX editor (e.g., Overleaf or TeXShop) if not already present.
4. Place any raw data CSV files in the data/ directory if separated from the script.

## Usage
### Running the Analysis
1. Open the Python script (e.g., analysis.py if created) containing the data processing and visualization code provided earlier.
2. Run the script to generate visualizations and the SQLite database:
   bash
   python analysis.py
   
   - Visualizations will display interactively or save to the output/ directory.
   - The database findex.db will be created in output/.


## Data Sources
- *WB_FINDEX_FIN14A*: World Bank Global Findex Database, 2024, reason for not having a mobile money account: mobile money agents are too far away.
- *WB_FINDEX_FIN14D*: World Bank Global Findex Database, 2024, reason for not having a mobile money account: don't have enough money to use a mobile money account.
- Data is embedded in the Python script via StringIO for this project; for real-world use, download from [World Bank Open Data](https://data.worldbank.org/).


## Acknowledgments
- World Bank Open Data for providing the datasets.
- The NDTA 631 course instructor for guidance and resources.

FINANCIAL BARRIERS IN THE DIGITAL AGE: WHY MANY STILL LACK MOBILE MONEY ACCOUNTS

