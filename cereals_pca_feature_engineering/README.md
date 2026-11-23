# Breakfast Cereals Rating Data Preparation with Principal Component Analysis

**Date:** November 9, 2026

## Project Overview

This project prepares 77 breakfast cereals (16 columns) for modeling their consumer rating. Using Python, pandas, Matplotlib, seaborn, and scikit-learn, the workflow removes the cereals with missing values, explores the manufacturers, shelves, rating, and nutritional variables, splits the data into training and test sets, scales the numeric predictors, applies principal component analysis (PCA), interprets the components, and encodes the categorical predictors.

The result is a training set of 59 rows and a test set of 15 rows, each with 14 numeric columns (6 principal components, 6 manufacturer columns, and 2 shelf columns) and no missing values. The project focuses on documenting every decision, fitting every step that learns from the data on the training set only to avoid data leakage, and communicating the limitations of a very small sample. It ends at the model-ready dataset; no model is trained.

## Business Context

Breakfast cereal manufacturers and retailers need to understand which nutritional characteristics are associated with better consumer ratings, both to evaluate existing products and to anticipate how a new product could be rated. The nutritional information of a cereal is described by many variables, and some of them move together. With only a few dozen products available, a model that receives all of them as separate inputs has little data to learn from and is prone to overfitting. Reducing the nutritional variables to a smaller set of uncorrelated components with PCA is one way to prepare the data for that model.

## Objectives

- Assess the structure, completeness, and descriptive statistics of the dataset.
- Handle the missing values with a documented decision.
- Explore how the products are distributed across manufacturers and shelves, and how the rating varies.
- Examine how the nutritional variables relate to the rating and to each other.
- Split the data into training and test sets before any step that learns from the data, to avoid data leakage.
- Scale the numeric predictors and apply PCA, selecting the number of components with a documented criterion.
- Interpret the principal components in terms of the original variables.
- Select and encode the categorical predictors, with a reason for each decision.
- Produce a training set and a test set that are ready for model training.

## Dataset

The analysis uses `Cereals.csv`, from the [Cereals datafile](https://dasl.datadescription.com/datafile/cereals/) of the Data and Story Library (DASL), a public collection of statistics datasets.

It contains 77 breakfast cereals and 16 columns:

- **Identifier:** `name`
- **Categorical:** `mfr` (manufacturer), `type` (cold or hot), `shelf` (display shelf, coded 1 to 3)
- **Nutritional values per serving:** `calories`, `protein`, `fat`, `sodium`, `fiber`, `carbo`, `sugars`, `potass`, `vitamins`, `weight`, `cups`
- **Target:** `rating`

The final dataset uses the 11 numeric nutritional variables (transformed into 6 principal components), `mfr`, and `shelf`. `name` and `type` are excluded, with the reasons documented in the notebook. In the original DASL file the missing nutrient values are coded as -1; in the file used here they appear as empty values.

The calculation of the `rating` is not documented, and 73 of the 74 cereals kept are cold cereals. The findings describe this dataset and should not be generalized to other cereals or markets.

## Tools and Libraries

- **Python:** programming language used for the analysis.
- **pandas and NumPy:** used to load, inspect, clean, and summarize the dataset.
- **Matplotlib and seaborn:** used to create bar plots, histograms, boxplots, line plots, and heatmaps.
- **scikit-learn:** used for the train and test split, standardization, PCA, and one-hot encoding.
- **Jupyter Notebook:** used to organize the analysis, document the process, and present the results.

## Project Structure

```text
cereals-pca-feature-engineering/
├── data/
│   ├── Cereals.csv
│   └── processed/
│       ├── cereals_train.csv
│       └── cereals_test.csv
├── notebooks/
│   └── cereals_pca_feature_engineering.ipynb
└── README.md
```

- `data/`: contains the dataset. The `processed/` folder is created when the notebook runs and holds the final training and test sets, each with the 14 predictors followed by the target `rating`.
- `notebooks/`: contains the Jupyter Notebook with the analysis.
- `README.md`: documents the project, its objectives, methodology, and findings.

## Analysis Workflow

The steps follow the numbered sections of the notebook.

1. **Load the data (section 5):** loaded the CSV file, defined the random seed and the test share, and documented the meaning of each column in a data dictionary.
2. **Inspect the dataset structure (section 6):** reviewed the first records, data types, non-null counts, and descriptive statistics of the numeric and text variables.
3. **Assess missing values (section 7):** counted the missing values, listed the three cereals that contain them, and removed those rows.
4. **Explore manufacturers and shelves (section 8):** counted the cereals by manufacturer and shelf, and combined both variables to see where each manufacturer places its products.
5. **Explore the rating (section 9):** summarized the distribution of the rating with statistics, counts per range of 10 points, and a bar plot, and listed the 10 best-rated cereals.
6. **Explore the nutritional variables (section 10):** compared the calories of each manufacturer, calculated the correlation of each numeric variable with the rating, and examined the correlations between the variables.
7. **Split into training and test sets (section 11):** divided the 74 cereals into 80% for training and 20% for testing, with a fixed random seed.
8. **Scale the numeric predictors and apply PCA (section 12):** compared the spread of the variables, standardized them with the training set, and compared PCA with and without scaling.
9. **Select the number of components (section 13):** reviewed the eigenvalue and the cumulative variance of each component, and kept the minimum number that explains more than 90% of the variance, with the Kaiser criterion as an alternative.
10. **Interpret the principal components (section 14):** reviewed the loadings, the cereals with the highest score in each component, and the correlation of each component with the rating.
11. **Select and encode the categorical predictors (section 15):** decided with evidence from the training set which categorical variables to keep, and applied one-hot encoding to them.
12. **Build and save the final dataset (section 16):** combined the components and the encoded variables, aligned the target, checked for missing values, matching columns, and data types, and saved both sets.
13. **Conclusions and next steps (section 17):** summarized findings, limitations, and next steps.

## Key Findings

- **Missing values:** four values were missing in three cereals (`potass`, `carbo`, and `sugars`). They were removed (3.9% of the rows), and two of those cereals were hot cereals, which leaves 73 cold cereals and 1 hot cereal.
- **Manufacturers and shelves:** Kelloggs and General Mills make up 60.8% of the 74 cereals, and shelf 3 holds 47.3% of them. Nabisco has 5 cereals and American Home Food Products a single one, so comparisons that involve them rest on very few cereals.
- **Rating:** it is negatively correlated with `sugars` (-0.756) and `calories` (-0.694) and positively with `fiber` (0.603), `protein` (0.467), and `potass` (0.416). All 5 Nabisco cereals are among the 10 best rated, and the best-rated cereal (All-Bran with Extra Fiber, 93.70) is far above the others.
- **Relationships between variables:** the numeric variables are only moderately correlated (the strongest pair is `fiber` and `potass`, 0.91), so PCA removes some redundancy but does not reduce the data to a very small number of components.
- **Scaling:** without scaling, the first component would explain 55.57% of the variance with a loading of 0.97 on `sodium`, so it would reflect the units and not the nutritional profile. After scaling, it explains 29.01% and the loading on `sodium` is -0.04.
- **PCA:** 6 components explain 90.41% of the variance of the training set, which reduces the 11 numeric variables to 6. The Kaiser criterion would keep only 3 components and 69.63% of the variance, so it was not used.
- **Component interpretation:** the first component reflects potassium, fiber, and protein (bran-like cereals), and the second reflects calories, serving weight, and sugars. The second component has the strongest correlation with the rating (-0.83), followed by the third (0.43) and the first (0.16).
- **Categorical predictors:** `name` (unique for every cereal) and `type` (58 cold cereals and 1 hot cereal in the training set) were excluded. `mfr` was kept as a nominal variable, and `shelf` was kept as a nominal and not an ordinal variable, because the rating is lowest on the middle shelf and similar on the other two.
- **Final dataset:** the training set has 59 cereals and the test set 15. Both have the same 14 numeric predictors (6 principal components, 6 manufacturer columns, and 2 shelf columns), no missing values, and `rating` as a separate target.

## Limitations

- **Small sample:** with 59 training cereals and 15 test cereals, the loadings, the number of components, and any future error measured on the test set are sensitive to the split.
- **Component selection:** the 90% threshold is a convention. The sixth component crosses it by a narrow margin (90.41%), so a different split could require a seventh.
- **Mostly cold cereals:** 73 of the 74 cereals kept are cold, so the findings do not describe hot cereals.
- **Unknown rating:** the calculation of the rating is not documented, so the associations found describe the data and are not explanations.
- **Exploration on all cereals:** the exploratory sections use all 74 cereals, including the test cereals. No transformation was fitted on them, and the decisions about the components and the categorical predictors use the training set only.
- **Small groups:** manufacturers `A` and `N` have 1 and 3 training cereals, so their mean ratings are very uncertain, and `A` is the reference category of the manufacturer encoding.
- **Shelf and sugar:** the lower rating on shelf 2 probably overlaps with the higher sugar content of those cereals, so `shelf` may add little to the numeric variables.
- **Interpretability:** PCA is a linear technique, and its components are harder to read than the original variables.
- **No model:** no model was trained, so the project does not show how the prepared data affects predictive performance.

## How to Run

1. Clone or download this repository.
2. Get `Cereals.csv` from the source listed in the Dataset section and save it in the `data/` directory.
3. Open `cereals_pca_feature_engineering.ipynb` in VS Code.
4. Make sure Python and the Jupyter extension are installed in VS Code.
5. Install the required libraries if they are not already available:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
```

6. Select the appropriate Python kernel and run the notebook cells in order, from top to bottom.

**Note:** the notebook loads the data with a relative path (`../data/Cereals.csv`) and saves the final sets in `../data/processed/`, so keep the folder structure as it is.

## Next Steps

- Train a regression model on the training set and evaluate it on the test set with error metrics, using cross-validation because the sample is small.
- Compare the model with the principal components against a model with the original scaled variables, to measure whether the reduction helps.
- Compare the model with and without `shelf` to measure its contribution.
- Repeat the selection of components with other splits to check how stable the number of components and the loadings are.
- Save the fitted scaler, PCA, and encoder so that new cereals can be transformed in the same way before they are given to the model.

---

*This project is a data preparation project for portfolio purposes. Findings and decisions are limited to the supplied dataset.*
