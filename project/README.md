
# Predicting Housing Prices - Kaggle Competition

Apply exploratory data analysis and implement various advanced supervised machine learning algorithms to predict neighborhood housing sale prices found in the sample test dataset. Compare the predicted models and results from these advanced supervised algorithms. Apply ensembled model to achieve better predictions, hence get good score in kaggle competition.
	
Part of the kaggle competition, two sample data sets are given with 80 attributes (variables) describing various aspects of the residential homes in Ames and Iowa cities. Training dataset contains sale price of the homes, and using this training data set, how accurately we can predict Sale Prices of the homes in the test dataset using preprocessing and thorough data analysis. Many developers used advanced learning algorithms - XGBoost, Lasso and Neural Network, to predict the sale prices in the kaggle competition and achieved better kaggle scores. Kaggle score is a measure to indicate accuracy and the quality of the algorithm. We have applied various exploratory analysis techniques and engineer the features before applying a few advanced supervised learning algorithms.

## Dataset Files:

* train.csv - the training data set with 1460 instances and 81 attributes including Sale Price, the target variable
* test.csv - the test data set with 1459 instances and 80 attributes excluding Sale Price

## List of features


    Id: row id
    SalePrice: the property’s sale price in dollars. This is the target variable to predict.
    MSSubClass: The building class
    MSZoning: The general zoning classification
    LotFrontage: Linear feet of street connected to property
    LotArea: Lot size in square feet
    Street: Type of road access
    Alley: Type of alley access
    LotShape: General shape of property
    LandContour: Flatness of the property
    Utilities: Type of utilities available
    LotConfig: Lot configuration
    LandSlope: Slope of property
    Neighborhood: Physical locations within Ames city limits
    Condition1: Proximity to main road or railroad
    Condition2: Proximity to main road or railroad (if a second is present)
    BldgType: Type of dwelling
    HouseStyle: Style of dwelling
    OverallQual: Overall material and finish quality
    OverallCond: Overall condition rating
    YearBuilt: Original construction date
    YearRemodAdd: Remodel date
    RoofStyle: Type of roof
    RoofMatl: Roof material
    Exterior1st: Exterior covering on house
    Exterior2nd: Exterior covering on house (if more than one material)
    MasVnrType: Masonry veneer type
    MasVnrArea: Masonry veneer area in square feet
    ExterQual: Exterior material quality
    ExterCond: Present condition of the material on the exterior
    Foundation: Type of foundation
    BsmtQual: Height of the basement
    BsmtCond: General condition of the basement
    BsmtExposure: Walkout or garden level basement walls
    BsmtFinType1: Quality of basement finished area
    BsmtFinSF1: Type 1 finished square feet
    BsmtFinType2: Quality of second finished area (if present)
    BsmtFinSF2: Type 2 finished square feet
    BsmtUnfSF: Unfinished square feet of basement area
    TotalBsmtSF: Total square feet of basement area
    Heating: Type of heating
    HeatingQC: Heating quality and condition
    CentralAir: Central air conditioning Electrical: Electrical system
    1stFlrSF: First Floor square feet
    2ndFlrSF: Second floor square feet
    LowQualFinSF: Low quality finished square feet (all floors)
    GrLivArea: Above grade (ground) living area square feet
    BsmtFullBath: Basement full bathrooms
    BsmtHalfBath: Basement half bathrooms
    FullBath: Full bathrooms above grade
    HalfBath: Half baths above grade
    Bedroom: Number of bedrooms above basement level
    Kitchen: Number of kitchens
    KitchenQual: Kitchen quality
    TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
    Functional: Home functionality rating
    Fireplaces: Number of fireplaces
    FireplaceQu: Fireplace quality
    GarageType: Garage location
    GarageYrBlt: Year garage was built
    GarageFinish: Interior finish of the garage
    GarageCars: Size of garage in car capacity
    GarageArea: Size of garage in square feet
    GarageQual: Garage quality
    GarageCond: Garage condition
    PavedDrive: Paved driveway
    WoodDeckSF: Wood deck area in square feet
    OpenPorchSF: Open porch area in square feet
    EnclosedPorch: Enclosed porch area in square feet
    3SsnPorch: Three season porch area in square feet
    ScreenPorch: Screen porch area in square feet
    PoolArea: Pool area in square feet
    PoolQC: Pool quality
    Fence: Fence quality
    MiscFeature: Miscellaneous feature not covered in other categories
    MiscVal: Dollar Value of miscellaneous feature
    MoSold: Month Sold
    YrSold: Year Sold
    SaleType: Type of sale
    SaleCondition: Condition of sale

# Processing Steps

We will do the following steps to analyze, model and evaluation of the sale prices of the test data:

    1. Exploratory Data Analysis:
        1.1. Analyze Numerical Variables
        1.2. Analyze Categorical Variables
        1.3. Analyze Outliers and Skewed Data
        1.4. Apply Feature Engineering
    2. Appply Various Machine Learning Algorithms
        2.1. Support Vector Machine (SVM) Algorithm
        2.2. Random Forest Algorithm
        2.3. Ridge Algorithm
        2.4. Lasso Algorithm
        2.5. Neural Network
        2.6. XGBoost
    3 Ensemble and Kaggle Submission


# Installation Instructions

	Operating Environment: Ubuntu 16.4

	Steps

	1. Install Jupytier Notebook (If not installed in the machine)
	   1.1. python -m pip install --upgrade pip
	   1.2. python -m pip install jupyter

	2. Install Anaconda (If not installed in the machine)
	   2.1. Dowload the installation file from the url: 
	   https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh
	   2.2. Run the following command in the terminal - bash ~/Downloads/Anaconda2-5.0.1-Linux-x86_64.sh

	3. The following packages need to be installed for the machine learning algorithms
	   3.1. pip install seaborn
	   3.2. pip install tensorflow
	   3.3. pip install xgboost

	4. Pull repository from Github
	   4.1. git@github.com:bigdata-i523/hid306.git

	5. Run the project code
	   5.1. The Project folder will have the source code, data and images folder.
	   5.2. If the project folder is under jupyter notebooks root directory, it can be accessed in Jupyter as following:
	   http://localhost:8888/tree/project/code
	   Note: (Assuming the folder name is 'project'. If the folder is different or placed under another folder replace the 'project'
	   with the appropriate directory structure and name)

	6. Juyper Notebook files - execution order	
	   6.1. file: 1.1_exploratory_analysis_numerical.ipynb
	   6.2. file: 1.2_exploratory_analysis_categorical.ipynb
	   6.3. file: 1.3_outlier_and_skewed_data_analysis.ipynb
	   6.4. file: 1.4_feature_engineering.ipynb
	   6.5. file: 2.1_algorithm_svm.ipynb
	   6.6. file: 2.2_algorithm_random_forest.ipynb
	   6.7. file: 2.3_algorithm_ridge.ipynb
	   6.8. file: 2.4_algorithm_lasso.ipynb
	   6.9. file: 2.5_algorithm_neural_network_tf.ipynb
	   6.10. file: 2.6_algorithm_xgboost.ipynb
	   6.12. file: 3_ensemble_kaggle_submission.ipynb
