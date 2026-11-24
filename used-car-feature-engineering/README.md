# Used Car Price Data Preparation and Feature Engineering

**Date:** November 4, 2026

## Project Overview

This project prepares 426,880 raw Craigslist used-car listings (26 columns) for price modeling. Using Python, pandas, Matplotlib, SciPy, scikit-learn, and category_encoders, the workflow selects the relevant variables, handles missing and invalid values with documented rules, splits the data into training and test sets, removes outliers with limits calculated on the training set only, transforms and scales the numeric variables, and encodes the categorical variables.

The result is a training set of 267,189 rows and a test set of 66,799 rows, each with 139 numeric columns and no missing values. The project focuses on documenting every cleaning decision, fitting every step that learns from the data on the training set only to avoid data leakage, and communicating the limitations of the data. It ends at the model-ready dataset; no model is trained.

## Business Context

Online marketplaces for used vehicles host hundreds of thousands of listings in which price, mileage, and vehicle attributes are typed in by the sellers themselves. Without a reference price, buyers cannot tell whether an ad is fairly priced, and the platform cannot flag listings that are overpriced, underpriced, or mistyped. A price-estimation model could provide that reference, but only if it is trained on data that has been cleaned and transformed consistently. The listings contain missing answers, placeholder prices, and impossible values, so the data must be prepared before any model is trained.

## Objectives

- Select the variables relevant to price estimation and document them in a data dictionary.
- Handle missing values and invalid records with explicit, justified rules.
- Split the data into training and test sets before any step that learns from the data, to avoid data leakage.
- Detect and treat outliers in price, mileage, and vehicle age with limits calculated on the training set only and applied to both sets.
- Reduce the skewness of the numeric variables, bring them to a comparable scale, and encode the categorical variables according to their cardinality.
- Produce a training set and a test set that are ready for model training.

## Dataset

The analysis uses `vehicles.csv`, from the [Used Cars Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) published on Kaggle by Austin Reese (Craigslist cars and trucks listings in the United States).

It contains 426,880 listings and 26 columns, posted between April 4 and May 5, 2021 (UTC), a span of 30 days. The final dataset uses 11 variables:

- **Target:** `price`
- **Numeric predictors:** `age` (derived from the model year `year`), `odometer`
- **Categorical predictors, low cardinality:** `manufacturer`, `fuel`, `title_status`, `transmission`, `type`, `state`
- **Categorical predictors, high cardinality:** `region`, `model`

The other 16 columns (identifiers, free text, redundant location fields, columns with 30% or more missing values, and `year` and `posting_date`, which are only used to calculate `age`) are removed and listed in the data dictionary of the notebook.

The prices are asking prices typed in by sellers, and the listings cover about one month. Therefore, the decisions describe that market snapshot and should not be generalized to other periods. The raw file is not included in the repository because of its size.

## Tools and Libraries

- **Python:** programming language used for the analysis.
- **pandas and NumPy:** used to load, inspect, clean, and summarize the dataset.
- **Matplotlib and SciPy:** used to create boxplots, histograms, and Q-Q plots.
- **scikit-learn:** used for the train and test split, the Yeo-Johnson transformation, Min-Max scaling, and one-hot encoding.
- **category_encoders:** used for the binary encoding of the high-cardinality variables.
- **Jupyter Notebook:** used to organize the analysis, document the process, and present the results.

## Project Structure

```text
used-car-price-feature-engineering/
├── data/
│   └── vehicles.csv
├── notebooks/
│   └── used_car_price_feature_engineering.ipynb
└── README.md
```

- `data/`: contains the dataset (not included in the repository; see How to Run).
- `notebooks/`: contains the Jupyter Notebook with the analysis.
- `README.md`: documents the project, its objectives, methodology, and findings.

## Analysis Workflow

The steps follow the numbered sections of the notebook.

1. **Load the data (section 5):** loaded the CSV file, defined the random seed and the test share, and documented the meaning of each column in a data dictionary.
2. **Inspect the dataset structure (section 6):** reviewed data types, non-null counts, and the range of posting dates.
3. **Select the columns (section 7):** removed identifiers, free text, and redundant location fields.
4. **Assess missing values (section 8):** dropped the columns with 30% or more missing values, dropped the rows with missing values in nearly complete columns, and filled the missing body type with an explicit `unknown` category.
5. **Create the vehicle age and apply validity rules (section 9):** converted the model year into age against a fixed reference year and removed prices of zero and default odometer readings, after reviewing the evidence for each rule.
6. **Split into training and test sets (section 10):** counted duplicated listings and divided the data into 80% for training and 20% for testing.
7. **Detect and remove outliers (section 11):** calculated the limits of the interquartile range rule on the training set only (price on a log10 scale), reviewed the flagged listings, and applied the same limits to both sets.
8. **Transform and scale the numeric features (section 12):** applied the Yeo-Johnson transformation and Min-Max scaling to `age` and `odometer`, fitted on the training set only, and compared the skewness before and after.
9. **Encode the categorical features (section 13):** applied one-hot encoding to the six low-cardinality variables and binary encoding to `model` and `region`, after counting the test categories not seen in training.
10. **Build the final dataset (section 14):** combined the numeric and encoded features, aligned the target, and checked for missing values, matching columns, and data types.
11. **Conclusions and next steps (section 15):** summarized the rows kept at each stage, the main findings, the limitations, and the next steps.

## Key Findings

- **Missing values:** six columns reached the 30% threshold and were dropped (`county` is empty), 37,276 listings (8.73%) with a missing value in a nearly complete column were removed, and the missing body type of 82,628 listings was labeled `unknown`.
- **Invalid records:** the validity rules removed 30,714 listings (7.88% of those left). 28,576 listings had a price of \$0, and 2,951 reported 2 miles or fewer on vehicles with a median age of 11 years, so those readings are default values.
- **Outliers:** the limits calculated on the training set are about \$1,044 to \$200,379 for price (on the log10 scale), up to 26 years for age, and up to about 280,000 miles for odometer. They flag 3.27%, 3.18%, and 0.78% of the training listings, and the removal took out 6.94% of both the training and the test set.
- **Numeric features:** the Yeo-Johnson transformation reduced the skewness of `age` from 0.785 to -0.041 and of `odometer` from 0.458 to -0.198.
- **Encoding:** binary encoding of `model` (16,407 categories in the training set) and `region` (404) creates 24 columns, instead of 16,809 with one-hot encoding.
- **Final dataset:** the final dataset keeps 78.2% of the raw listings (333,988): 267,189 for training and 66,799 for testing. Both sets have 139 columns (2 scaled numeric, 113 one-hot, and 24 binary-encoded), no missing values, and the same columns.

## Limitations

- **Asking prices and one period:** the prices are asking prices from about one month in 2021, so the prepared data describes that market snapshot only.
- **Outlier removal:** the interquartile range rule removes real vehicles. Among the most frequent prices below the lower limit, \$1 (1,286 listings) and \$99 (639) are placeholders, but \$1,000 (733) and \$500 (593) can be real asking prices for cheap old cars. The upper limits also remove luxury cars, classic cars, ordinary older vehicles, and vehicles with very high mileage, so the data is best suited to the mainstream used-car market. The same limits are applied to the test set, so the evaluation of a future model describes that segment only.
- **Remaining choices:** the multiplier of 1.5, the log scale for price, and the cutoff of 2 miles for default odometer readings are supported by the evidence in the notebook, but other reasonable choices would give a slightly different dataset.
- **Default mileage above 2 miles:** readings above 2 miles may also contain defaults (for example, 331 listings report exactly 10 miles), which are kept because the evidence does not separate them from low-mileage vehicles.
- **Duplicated listings:** 44,773 listings (12.48% of the cleaned dataset) are exact duplicates and were not removed, so reposted ads can appear in both the training and the test set and make a future evaluation too optimistic.
- **Cleaning rules fitted on all data:** the missing-value thresholds were applied to the full dataset before the split.
- **High-cardinality encoding:** model names are typed by sellers, so the binary encoding of `model` is a compromise. 1,881 test rows (2.82%) have a model not seen in training, and they share one reserved code.
- **No model:** no model was trained, so the project does not show how the prepared data affects predictive performance.

## How to Run

1. Clone or download this repository.
2. Get `vehicles.csv` from the source listed in the Dataset section and save it in the `data/` directory.
3. Open `used_car_price_feature_engineering.ipynb` in VS Code.
4. Make sure Python and the Jupyter extension are installed in VS Code.
5. Install the required libraries if they are not already available:

```bash
python -m pip install pandas numpy matplotlib scipy scikit-learn category_encoders
```

6. Select the appropriate Python kernel and run the notebook cells in order, from top to bottom.

**Note:** the notebook loads the data with a relative path (`../data/vehicles.csv`), so keep the folder structure as it is.

## Next Steps

- Train and compare regression models on the training set, and evaluate them on the test set with error metrics.
- Review the duplicated listings and decide whether to remove them or to keep each group together in one set.
- Reduce the number of `model` categories, for example by grouping rare names, and test whether it improves the encoding.
- Test a transformation of the target `price` at the modeling stage.

---

*This project is a data preparation project for portfolio purposes. Findings and decisions are limited to the supplied dataset.*
