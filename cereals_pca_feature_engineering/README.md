# Breakfast Cereals Rating Data Preparation with Principal Component Analysis

Data preparation of 77 breakfast cereals to model their consumer rating: missing-value handling, exploratory analysis, a leakage-free train/test split, scaling, principal component analysis (PCA) and encoding of the categorical predictors. The project ends with a training set and a test set that are ready for model training.

## Business context

Breakfast cereal manufacturers and retailers need to understand which nutritional characteristics are associated with better consumer ratings, both to evaluate existing products and to anticipate how a new product could be rated. The nutritional information of a cereal is described by many variables, and some of them move together. With only a few dozen products, a model that receives all of them as separate inputs has little data to learn from and is prone to overfitting. Reducing the nutritional variables to a smaller set of uncorrelated components with PCA is one way to prepare the data for that model.

## Project objective

The notebook prepares the cereals dataset for modeling the product rating. It covers the following steps:

- assess the structure, completeness and descriptive statistics of the dataset;
- handle the missing values with a documented decision;
- explore how the products are distributed across manufacturers and shelves, and how the rating varies;
- examine how the nutritional variables relate to the rating and to each other;
- split the data into training and test sets before any step that learns from the data, to avoid data leakage;
- scale the numeric predictors and apply PCA, selecting the number of components with a documented criterion;
- interpret the principal components in terms of the original variables;
- select and encode the categorical predictors, with a reason for each decision; and
- produce a training set and a test set that are ready for model training.

The project is data preparation. It does not train a model or estimate its performance.

## Dataset

The file used is `Cereals.csv`, from the [Cereals datafile](https://dasl.datadescription.com/datafile/cereals/) of the Data and Story Library (DASL). It contains the nutritional information and the shelf location of 77 breakfast cereals from 7 manufacturers, in 16 columns. The target is `rating`; the other columns are nutritional values per serving (`calories`, `protein`, `fat`, `sodium`, `fiber`, `carbo`, `sugars`, `potass`, `vitamins`, `weight`, `cups`), the manufacturer (`mfr`), the cereal type (`type`), the shelf (`shelf`) and the name of the cereal (`name`). The data dictionary is in section 5 of the notebook.

In the original DASL file the missing nutrient values are coded as -1; in the file used here they appear as empty values.

## Approach

1. **Inspect and clean.** Review the structure and the descriptive statistics, and remove the 3 cereals with missing values (3.9% of the rows).
2. **Explore.** Count the cereals by manufacturer and shelf, describe the rating, and compare the nutritional variables with the rating and with each other.
3. **Split.** Divide the 74 remaining cereals into 59 for training and 15 for testing, with a fixed random seed (42). The scaler, the PCA and the encoder are fitted on the training set only and then applied to the test set.
4. **Scale and apply PCA.** Standardize the 11 numeric predictors, compare PCA with and without scaling, and keep the minimum number of components that explains more than 90% of the variance of the training set. The Kaiser criterion is reported as an alternative.
5. **Interpret the components.** Review the loadings, the cereals with the highest scores, and the correlation of each component with the rating.
6. **Encode the categorical predictors.** Exclude `name` and `type` with evidence from the training set, keep `mfr` and `shelf` as nominal variables, and encode them with one-hot encoding.
7. **Build and save the final sets.** Combine the components and the encoded variables, check the result, and save both sets.

## Key findings

- **Missing values:** 4 values were missing in 3 cereals (`potass`, `carbo` and `sugars`). Two of those cereals were hot cereals, so the cleaned data contains 73 cold cereals and 1 hot cereal.
- **Rating:** it is negatively correlated with `sugars` (-0.756) and `calories` (-0.694) and positively with `fiber` (0.603), `protein` (0.467) and `potass` (0.416). All 5 Nabisco cereals are among the 10 best rated.
- **Relationships between variables:** the numeric variables are only moderately correlated (the strongest pair is `fiber` and `potass`, 0.91), so PCA removes some redundancy but does not reduce the data to a very small number of components.
- **PCA:** after scaling, 6 components explain 90.41% of the variance of the training set, which reduces the 11 numeric variables to 6. Without scaling, the first component would be built almost only from `sodium` (loading of 0.97) because of its units. The first component reflects potassium, fiber and protein (bran-like cereals), and the second reflects calories, serving weight and sugars. The second component has the strongest correlation with the rating (-0.83).
- **Categorical predictors:** `name` (unique for every cereal) and `type` (58 cold cereals and 1 hot cereal in the training set) were excluded. `mfr` was kept as a nominal variable, and `shelf` was kept as a nominal and not an ordinal variable, because the rating is lowest on the middle shelf and similar on the other two.
- **Final datasets:** the training set has 59 cereals and the test set 15. Each set has 14 predictors (6 principal components, 6 manufacturer columns and 2 shelf columns), with no missing values, and `rating` as the target.

## Limitations

- **Small sample:** with 59 training cereals and 15 test cereals, the loadings, the number of components and any future error measured on the test set are sensitive to the split. The sixth component crosses the 90% threshold by a narrow margin.
- **Mostly cold cereals:** 73 of the 74 cereals kept are cold, so the findings do not describe hot cereals.
- **Unknown rating:** the calculation of the `rating` is not documented, so the associations found describe the data and are not explanations.
- **Exploration on all cereals:** the exploratory sections use all 74 cereals, including the test cereals. No transformation was fitted on them, and every decision about the components and the categorical predictors uses the training set only.
- **Small groups:** manufacturers `A` and `N` have 1 and 3 training cereals, so their mean ratings are very uncertain.
- **Shelf and sugar:** the lower rating on shelf 2 probably overlaps with the higher sugar content of those cereals, so `shelf` may add little to the numeric variables.
- **Interpretability:** PCA is a linear technique, and its components are harder to read than the original variables.

## Project structure

```
.
├── data/
│   ├── Cereals.csv
│   └── processed/
│       ├── cereals_train.csv
│       └── cereals_test.csv
├── notebooks/
│   └── cereals_pca_feature_engineering.ipynb
└── README.md
```

`Cereals.csv` must be downloaded from DASL and saved in the `data` folder before running the notebook. The files in `data/processed` are created when the notebook runs; each one has the 14 predictors followed by the target `rating`.

## How to run

1. Install Python 3 and the libraries used in the notebook: pandas (2.1 or later), NumPy, Matplotlib, seaborn, scikit-learn and Jupyter.
2. Save `Cereals.csv` in the `data` folder.
3. Open `notebooks/cereals_pca_feature_engineering.ipynb` and run all the cells from the top. The paths are relative to the `notebooks` folder, and the random seed and the test size are defined in section 5.

## Next steps

- Train a regression model on the training set and evaluate it on the test set with error metrics, using cross-validation because the sample is small.
- Compare the model with the principal components against a model with the original scaled variables, to measure whether the reduction helps.
- Compare the model with and without `shelf` to measure its contribution.
- Repeat the selection of components with other splits to check how stable the number of components and the loadings are.
- Save the fitted scaler, PCA and encoder so that new cereals can be transformed in the same way before they are given to the model.

## Tools

Python, pandas, NumPy, Matplotlib, seaborn and scikit-learn.

---

This project prepares the data for modeling for portfolio purposes. It does not train a model, and it makes no claim about predictive performance. Its conclusions are limited to the supplied dataset.
