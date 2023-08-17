import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

gas = pd.read_csv("gas_prices.csv", encoding='utf-8')
#get columns for each country
countries = ['India', 'Japan', 'Russia', 'China', 'US', 'World', 'Canada']

#plot the predictions for each country
for country in countries:
    ################################  Polynomial features #################################
    if country == 'Japan' or country == 'Russia' or country == 'China':
        #we did many runs and India, US and World are better fitted with kNN so we are not going to graph
        #for polyfeatures and we are going to graph them later with kNN
        country_data = gas[[gas.columns[0], country]].dropna()
        X = country_data[[gas.columns[0]]]
        y = country_data[country]

        
        #ideal test size values for each country
        if country == 'Japan':
            size = 0.4 
        if country == 'Russia':
            size = 0.3
        if country == 'China':
            size = 0.5

            
        #create pipeline for preprocessing and model fitting
        pipeline = make_pipeline(PolynomialFeatures(degree=5), LinearRegression())


        #split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
        
        #fit the model
        pipeline.fit(X_train, y_train)

        #define range for fuel cost from 2006 to 2100
        X_range = np.arange(2006, 2100,0.1).reshape(-1, 1)
        # Smaller plot range 
        X_range_small = np.arange(2006, 2022,0.1).reshape(-1, 1)    
    
        #long term prediction
        y_pred = pipeline.predict(X_range)
        # Short term prediction
        y_pred_small = pipeline.predict(X_range_small)   

        # Calculate R-squared score for the model on the test set
        r_squared = pipeline.score(X_test, y_test)
        print(f'Model Score for {country}: {r_squared:.3f}')

        plt.figure(figsize=(8, 5))
        
        # Create a new plot for the current country
        # plt.subplot(121)
        # plt.scatter(X, y, label='Actual Data')
        # plt.plot(X_range, y_pred, label='Predictions', color='red')
        # plt.xlabel('Year')
        # plt.ylabel('Fuel Cost (USD/Liter)')
        # plt.title(f'Fuel Cost Predictions for {country} 2006-2100')
        # plt.legend()
        # plt.grid(True)
        
        # Plot the smaller graph for the specific country
        # plt.subplot(122)
        plt.scatter(X, y, label='Actual Data')
        plt.plot(X_range_small, y_pred_small, 'r-', label='Predictions')
        plt.xlabel('Year')
        plt.title(f'Fuel Cost Predictions for {country} 2006-2022 (Model Score = {r_squared:.3f})')
        plt.legend()
        plt.grid(True)

        # Show the plots for the specific country
        plt.tight_layout()
        plt.savefig(f'plots\cost\{country}_cost_plot.png')
        # plt.show()

    ################################  kNN #################################

#plot the predictions for India, US, and World
    else:
        country_data = gas[[gas.columns[0], country]].dropna()
        X = country_data[[gas.columns[0]]]
        y = country_data[country]

        #china only has values starting from 2009
        if country == 'China':
            X_range = np.arange(2009, 2101, 0.1).reshape(-1, 1)
            X_range_small = np.arange(2009, 2022,0.1).reshape(-1, 1)
            pipeline = make_pipeline(KNeighborsRegressor(n_neighbors=3))
        else:
            X_range = np.arange(2006, 2101,0.1).reshape(-1, 1)
            X_range_small = np.arange(2006, 2022,0.1).reshape(-1, 1)
            pipeline = make_pipeline(KNeighborsRegressor(n_neighbors=3))

        #ideal test size values for each country
        if country == 'India':
            size = 0.3
        if country == 'US' or country == 'World':
            size = 0.2

            
        #model + training + print r^2 value
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_range)
        y_pred_small = pipeline.predict(X_range_small)

        r_squared = pipeline.score(X_test, y_test)
        print(f'Model Score for {country}: {r_squared:.3f}')

        plt.figure(figsize=(8, 5))

        # plt.subplot(121)
        # plt.scatter(X, y, label='Actual Data')
        # plt.plot(X_range, y_pred, label='Predictions', color='red')
        # plt.xlabel('Year')
        # plt.ylabel('Fuel Cost (USD/Liter)')
        # plt.title(f'Fuel Cost Predictions for {country} 2006-2100')
        # plt.legend()
        # plt.grid(True)

        # plt.subplot(122)
        plt.scatter(X, y, label='Actual Data')
        plt.plot(X_range_small, y_pred_small, 'r-', label='Predictions')
        plt.xlabel('Year')
        plt.title(f'Gas Cost Fit for {country} 2009-2022 (Model Score = {r_squared:.3f})')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'plots\cost\{country}_cost_plot.png')
        # plt.show()


#     ################################  Random Forest #################################

# for country in countries:
#     country_data = gas[[gas.columns[0], country]].dropna()
#     X = country_data[[gas.columns[0]]]
#     y = country_data[country]

#     if country == 'China':
#         X_range = np.arange(2009, 2101).reshape(-1, 1)
#         X_range_small = np.arange(2009, 2022).reshape(-1, 1)
#     else:
#         X_range = np.arange(2006, 2101).reshape(-1, 1)
#         X_range_small = np.arange(2006, 2022).reshape(-1, 1)

#     pipeline = make_pipeline(RandomForestRegressor(n_estimators=500,max_depth = 20, random_state=42))
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     pipeline.fit(X_train, y_train)

#     y_pred = pipeline.predict(X_range)
#     y_pred_small = pipeline.predict(X_range_small)

#     r_squared = pipeline.score(X_test, y_test)
#     print(f'R-squared score for {country}: {r_squared:.3f}')

#     plt.figure(figsize=(12, 5))

#     plt.subplot(121)
#     plt.scatter(X, y, label='Actual Data')
#     plt.plot(X_range, y_pred, label='Predictions', color='red')
#     plt.xlabel('Year')
#     plt.ylabel('Fuel Cost (USD/Liter)')
#     plt.title(f'Fuel Cost Predictions for {country} 2006-2100')
#     plt.legend()
#     plt.grid(True)

#     plt.subplot(122)
#     plt.scatter(X, y, label='Actual Data')
#     plt.plot(X_range_small, y_pred_small, 'r-', label='Predictions')
#     plt.xlabel('Year')
#     plt.title(f'Fuel Cost Predictions for {country} 2009-2022 (R-squared = {r_squared:.3f})')
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.show()

#   ################################  MLP progressor #################################
# for country in countries:
#     country_data = gas[[gas.columns[0], country]].dropna()
#     X = country_data[[gas.columns[0]]]
#     y = country_data[country]

#     if country == 'China':
#         X_range = np.arange(2009, 2101).reshape(-1, 1)
#         X_range_small = np.arange(2009, 2022).reshape(-1, 1)
#     else:
#         X_range = np.arange(2006, 2101).reshape(-1, 1)
#         X_range_small = np.arange(2006, 2022).reshape(-1, 1)

#     pipeline = make_pipeline(MLPRegressor(hidden_layer_sizes=(500,200), activation='logistic', solver='lbfgs'))
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     pipeline.fit(X_train, y_train)

#     y_pred = pipeline.predict(X_range)
#     y_pred_small = pipeline.predict(X_range_small)

#     r_squared = pipeline.score(X_test, y_test)
#     print(f'R-squared score for {country}: {r_squared:.3f}')

#     plt.figure(figsize=(12, 5))

#     plt.subplot(121)
#     plt.scatter(X, y, label='Actual Data')
#     plt.plot(X_range, y_pred, label='Predictions', color='red')
#     plt.xlabel('Year')
#     plt.ylabel('Fuel Cost (USD/Liter)')
#     plt.title(f'Fuel Cost Predictions for {country} 2006-2100')
#     plt.legend()
#     plt.grid(True)

#     plt.subplot(122)
#     plt.scatter(X, y, label='Actual Data')
#     plt.plot(X_range_small, y_pred_small, 'r-', label='Predictions')
#     plt.xlabel('Year')
#     plt.title(f'Fuel Cost Predictions for {country} 2009-2022 (R-squared = {r_squared:.3f})')
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.show()

#################################### FOR COMBINING PLOTS FOR THE REPORT #####################################
# gas = pd.read_csv("gas_prices.csv", encoding='utf-8')
# countries = ['Japan', 'Russia', 'China']

# # Create a single plot for countries using Polynomial Features
# plt.figure(figsize=(8, 5))

# # Plot the predictions for each country
# for country in countries:
#     country_data = gas[[gas.columns[0], country]].dropna()
#     X = country_data[[gas.columns[0]]]
#     y = country_data[country]

#     # If country is in the Polynomial Features group
#     if country in ['Japan', 'Russia', 'China']:
#         pipeline = make_pipeline(PolynomialFeatures(degree=5), LinearRegression())
#         X_range = np.arange(2006, 2100, 0.1).reshape(-1, 1)
#         X_range_small = np.arange(2006, 2022, 0.1).reshape(-1, 1)

#         # Ideal test size values for each country
#         size = 0.4 if country == 'Japan' else 0.3 if country == 'Russia' else 0.5

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
# plt.title('Fuel Cost Predictions for Countries using Polynomial Features')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig('plots/cost/predictions_poly_features.png')
# plt.show()
