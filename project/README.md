
# Predicting Housing Prices - Kaggle Competition

Apply exploratory data analysis and linear regression, a supervised machine learning algorithm, to predict Sale Prices of list of homes found in the sample test data using training data having Sale Prices in the neighborhood area. Compare the predicted model and results with various other supervised algorithms.

Two sample data sets are given with 79 attributes describing various aspects of the residential homes in Ames and Iowa cities. Training data set contains Sale Price of the homes, using the training data set, how accurately we can predict Sale Prices of the homes in the test data set using thorough exploratory data analysis.

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




