import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("IEA-EV-dataEV-salesCarsHistorical.csv")
top_countries = ["World", "USA", "China", "Japan", "India", "Canada", "Russia"]

#################################  Polynomial features #################################
for country in top_countries:
    # Select country and keep only year and value
    data = df[df['region'] == country]
    columns_to_keep = ['year', 'value'] 
    data = data.drop(columns=set(df.columns) - set(columns_to_keep))

    # Sum by year to get the total ev's sold for that year
    data = data.groupby('year')['value'].sum()
    data = data.reset_index()
    data.rename(columns={'value': 'value'}, inplace=True)

    X = data[['year']]
    y = data['value']

    if country == "USA" or country == "China" or country == "World":
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # poly = PolynomialFeatures(degree=3, include_bias=False)
    # X_train_poly = poly.fit_transform(X_train)
    # X_valid_poly = poly.transform(X_valid)
    
    # X_range = np.arange(2010, 2023, 1).reshape(-1, 1)

    # model = LinearRegression()
    # model.fit(X_train_poly, y_train)
    
    # r_squared = model.score(X_valid_poly, y_valid)
    # y_predicted = model.predict(poly.fit_transform(X_range))

    # max_year = 2100
    # X_range_extended = np.arange(2022, max_year+1, 1).reshape(-1, 1)
    # X_range_extended_poly = poly.transform(X_range_extended)
    # y_predicted_extended = model.predict(X_range_extended_poly)

    pipeline = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    pipeline.fit(X_train.values, y_train.values)

    # Define range for fuel cost from 2006 to 2100
    X_range_extended = np.arange(2010, 2100).reshape(-1, 1)
    # Smaller plot range 
    X_range = np.arange(2010, 2023).reshape(-1, 1)    
   
    # Long term prediction
    y_predicted_extended = pipeline.predict(X_range_extended)
    # Short term prediction
    y_predicted = pipeline.predict(X_range)  

    # Calculate R-squared score for the model on the test set
    r_squared = pipeline.score(X_valid.values, y_valid.values)
    print(f'Model Score for {country}: {r_squared:.3f}')

    plt.figure(figsize=(8, 5))

    # plt.subplot(121)
    plt.scatter(X['year'], y, label='Actual Data')
    plt.plot(X_range, y_predicted, label='Predictions', color='red')
    plt.xlabel('Year')
    plt.ylabel('Total Electric Vehicles Sold')
    plt.title(f'Electric Vehicles Sold in {country} 2010-2022 (R-squared = {r_squared:.3f})')
    plt.legend()
    plt.grid(True)

    # plt.subplot(122)
    # plt.scatter(X['year'], y, label='Actual Data')
    # plt.plot(X_range_extended, y_predicted_extended, 'r-', label='Predictions')
    # plt.xlabel('Year')
    # plt.ylabel('Total Electric Vehicles Sold')
    # plt.title(f'Electric Vehicles sold in {country} 2010-2100')
    plt.legend()
    plt.grid(True)

    plt.ticklabel_format(useOffset=False, style='plain')    # comment out when creating subplots
    plt.tight_layout()
    plt.savefig(f'plots\EVSales\{country}_EV_plot.png')
    # plt.show()

#     ################################ kNN Regressor ####################################

# for country in top_countries:
#     data = df[df['region'] == country]
#     columns_to_keep = ['year', 'value'] 
#     data = data.drop(columns=set(df.columns) - set(columns_to_keep))

#     # Sum by year to get the total ev's sold for that year
#     data = data.groupby('year')['value'].sum()
#     data = data.reset_index()
#     data.rename(columns={'value': 'value'}, inplace=True)

#     X = data[['year']]
#     y = data['value']


#     X_range = np.arange(2010, 2101).reshape(-1, 1)
#     X_range_small = np.arange(2010, 2023).reshape(-1, 1)


#     pipeline = make_pipeline(KNeighborsRegressor(n_neighbors=2))
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
#     pipeline.fit(X_train.values, y_train.values)
    
#     y_pred = pipeline.predict(X_range)
#     y_pred_small = pipeline.predict(X_range_small)

#     r_squared = pipeline.score(X_test.values, y_test.values)
#     print(f'Model Score for {country}: {r_squared:.3f}')

#     plt.scatter(X, y, label='Actual Data')
#     plt.plot(X_range_small, y_pred_small, 'r-', label='kNN Predictions')
#     plt.xlabel('Year')
#     plt.ylabel('Total Electric Vehicles Sold')
#     plt.title(f'Electric Vehicles Sold in {country} 2010-2022 (Model Score = {r_squared:.3f})')
#     plt.legend()
#     plt.grid(True)
#     plt.ticklabel_format(useOffset=False, style='plain')
#     plt.savefig(f'plots\EVSales\{country}_EV_plot.png')
#     plt.show()

    ################################### Random Forest ###################################

# for country in top_countries:
#     data = df[df['region'] == country]
#     columns_to_keep = ['year', 'value'] 
#     data = data.drop(columns=set(df.columns) - set(columns_to_keep))

#     # Sum by year to get the total ev's sold for that year
#     data = data.groupby('year')['value'].sum()
#     data = data.reset_index()
#     data.rename(columns={'value': 'value'}, inplace=True)

#     X = data[['year']]
#     y = data['value']

#     X_range = np.arange(2010, 2101).reshape(-1, 1)
#     X_range_small = np.arange(2010, 2023).reshape(-1, 1)

#     pipeline = make_pipeline(RandomForestRegressor(n_estimators=500,max_depth = 20, random_state=42))
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     pipeline.fit(X_train.values, y_train.values)

#     y_pred = pipeline.predict(X_range)
#     y_pred_small = pipeline.predict(X_range_small)

#     r_squared = pipeline.score(X_test.values, y_test.values)
#     print(f'R-squared score for {country}: {r_squared:.3f}')

#     plt.figure(figsize=(12, 5))

#     plt.subplot(122)
#     plt.scatter(X, y, label='Actual Data')
#     plt.plot(X_range, y_pred, label='Random Forest Predictions', color='red')
#     plt.xlabel('Year')
#     plt.ylabel('Total Electric Vehicles Sold')
#     plt.title(f'Electric Vehicles Sold in {country} 2010-2100')
#     plt.legend()
#     plt.grid(True)

#     plt.subplot(121)
#     plt.scatter(X, y, label='Actual Data')
#     plt.plot(X_range_small, y_pred_small, 'r-', label='Random Forest Predictions')
#     plt.xlabel('Year')
#     plt.ylabel('Total Electric Vehicles Sold')
#     plt.title(f'Electric Vehicles sold in {country} 2010-2022 (R-squared = {r_squared:.3f})')
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.show()

# ############################# FOR THE REPORT COMBINING PLOTS TOGETHER ##########################
# # need to manually change the countries you wish to have on the plots
# countries = ['India', 'US', 'World']

# # Create a single plot for countries using KNeighborsRegressor
# plt.figure(figsize=(8, 5))

# # Plot the predictions for each country
# for country in countries:
#     country_data = gas[[gas.columns[0], country]].dropna()
#     X = country_data[[gas.columns[0]]]
#     y = country_data[country]
#     # If country is in the kNN group
#     if country in ['India', 'US', 'World']:
#         pipeline = make_pipeline(KNeighborsRegressor(n_neighbors=3))
#         X_range = np.arange(2009, 2101, 0.1).reshape(-1, 1)
#         X_range_small = np.arange(2009, 2022, 0.1).reshape(-1, 1)

#         # Ideal test size values for each country
#         if country == 'India':
#             size = 0.3
#         if country == 'US' or country == 'World':
#             size = 0.2

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
#         pipeline.fit(X_train, y_train)

#         y_pred = pipeline.predict(X_range)
#         y_pred_small = pipeline.predict(X_range_small)

#         r_squared = pipeline.score(X_test, y_test)
#         print(f'Model Score for {country}: {r_squared:.3f}')

#         plt.scatter(X, y, label=f'Actual Data ({country})')
#         plt.plot(X_range_small, y_pred_small, label=f'Predictions ({country})')

# plt.xlabel('Year')
# plt.ylabel('Fuel Cost (USD/Liter)')
# plt.title('Fuel Cost Predictions for Countries using KNeighborsRegressor')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig('plots/cost/predictions_knn.png')
# plt.show()
