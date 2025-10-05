# Project SkySight(A data driven Flight Difficulty Score)
## Built for United Airlines Hackathon (SkyHack)

## 1. Overview

This project addresses a key operational challenge for United Airlines at Chicago O’Hare International Airport (ORD): the inconsistent and reactive process of identifying high-difficulty flights. The current system relies on personal experience, which is not scalable and can lead to inefficient resource allocation and preventable delays.

Project Skysight replaces this manual approach with a robust, data-driven framework. By analyzing two weeks of flight, passenger, and baggage data, this project develops a Flight Difficulty Score that systematically quantifies the operational complexity of every flight before departure. This enables proactive planning, optimized resource allocation, and a more resilient ground operation.

The project is delivered in two parts:

1. An explainable, heuristic score based on our key findings from Exploratory Data Analysis (EDA). This is the primary, recommended solution.

2. A proof-of-concept Machine Learning model that predicts departure delays, serving as a powerful tool for future development.

## 2. Project Structure

```
SkyHack
|__Codes
|   |__build_score_v2.py (main script 3)
|   |__clean_scripts.py (main script)
|   |__diff_dest.py (personal helper)
|   |__generate_final_plots.py (personal helper)
|   |__run_eda_v2.py (main script 2)
|   |__visualise_eda.py (personal helper)
|   |__why.py (personal helper)
|__Dataset
|   |__Airports+Data.csv
|   |__Bag+Level+Data.csv
|   |__Flight+Level+Data.csv
|   |__PNR+Flight_Level_Data.csv
|   |__PNR+Remark_Level_Data.csv
|__output
|   |__test_ridhayneerajnathoo.csv (Final csv file for submission)
|   |__ other output files
|__Advanced_ML_PoC
   |__final_model.py (script 3)
   |__finetune_ml_model.py (script 2)
   |__train_model.py (script 1)
```

## 3. Installation and Workflow

### Installation
Clone the repository and install the required Python libraries:
```
pip install pandas numpy matplotlib seaborn lightgbm optuna "optuna-integration[lightgbm]"
```

### How to run the project
This project is designed to be run as a sequence of scripts. Execute them in the following order.

#### Part 1: The Heuristic Flight Difficulty Score (Main Solution)

1. Run the Cleaning Script: This script takes all the raw CSV files from the Dataset/ folder, performs a comprehensive cleaning process, and creates a single master file named "cleaned_flight_features_v2.csv" in the output/ folder.
   
    ``` python Codes/clean_scripts.py ```

2. Run the EDA Script: This script loads the clean data, performs the Exploratory Data Analysis, and saves the findings to eda_results_v2.txt and a series of plots.

    ```python Codes/run_eda_v2.py```

3. Run the Scoring Script: This script uses the clean data and EDA insights to calculate the final Flight Difficulty Score for each flight, saving the results to "test_ridhayneerajnathoo.csv".

    ```python Codes/build_score_v2.py```

4. The remaining scripts are optional and are there for visualisations and interpreting the final results.

#### Part 2: Predictive Modeling (Future Scope)

These scripts are located in SkyHack/Advanced_ML_PoC/ directory.

1. Train a Baseline Model (Optional):
   
   ```python Advanced_ML_PoC/train_model.py```

2. Fine-Tune the Model (Optional):

    ```python Advanced_ML_PoC/finetune_ml_model.py```

3. Generate Final Predictions: This script uses the best fine-tuned parameters to train a final model and generate a sample prediction report.

    ```python Advanced_ML_PoC/final_model.py```

## 4. Key Findings and Solution

### Exploratory Data Analysis (EDA)
Our analysis of the cleaned data revealed several key drivers of flight difficulty:

1. Special Service Requests (SSRs): The most consistent predictor of delays. Flights with high SSRs are always more delayed.

2. Ground Time Cushion: Over 8% of flights are scheduled with a negative time buffer, a major structural risk.

3. Hot Transfer Bags: 9% of an average flight's bags are time-sensitive "Hot Transfers," a key indicator of baggage pressure.

4. Passenger Load (The Myth): We proved that there is no correlation between how full a flight is and its likelihood of delay.

### The Flight Difficulty Score
Our primary solution is an explainable, weighted score that ranks each flight's difficulty based on the drivers identified in the EDA. This provides a clear, daily priority list for the operations team.

### Predictive Modeling (Future Scope)
As a proof-of-concept, we also developed a machine learning model that predicts the actual departure delay in minutes. After fine-tuning, the model achieved a Mean Absolute Error of ~21 minutes, demonstrating a powerful capability for future development. Interestingly, the ML model confirmed that the features we selected for our heuristic score were indeed the most important for predicting delays.

## Final Outputs
Running the main workflow will produce the following key files:

1. ```cleaned_flight_features_v2.csv```: The final, clean master dataset used for all analysis.
2. ```eda_results_v2.txt```: A text file summarizing the key findings from the EDA.
3. ```test_ridhayneerajnathoo.csv```: The final deliverable, containing the ranked and classified difficulty score for every flight.
4. ```.png```image files: Various plots visualizing the analysis. 