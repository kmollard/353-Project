import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# import statsmodels.api as sm
# from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures

#load data and only keep years between 1992 and 2019
fuel_consumption = pd.read_csv("Fuel_production_vs_consumption.csv", encoding='utf-8')

#filter the data for Russia and USSR (we are going to use this later)
#we are making a copy so we don;t modify the original dataframe (since we need it)
russia_ussr_data = fuel_consumption[((fuel_consumption['Entity'] == 'Russia') | (fuel_consumption['Entity'] == 'Former U.S.S.R.')) & 
                                  (fuel_consumption['Year'] >= 1980) & (fuel_consumption['Year'] <= 2019)].copy()

#convert 'Oil consumption(m)' from cubic meters to barrels (1 cubic meter = 6.28981 barrels) for Russia/USSR data
russia_ussr_data['Oil consumption(bbl)'] = russia_ussr_data['Oil consumption(m)'] * 6.28981

#load data and only keep years between 1992 and 2019
fuel_consumption = fuel_consumption[(fuel_consumption['Year'] >= 1992) & (fuel_consumption['Year'] <= 2019)]

#convert 'Oil consumption(m)' from cubic meters to barrels (1 cubic meter = 6.28981 barrels) for rest of the data
fuel_consumption['Oil consumption(bbl)'] = fuel_consumption['Oil consumption(m)'] * 6.28981

#group the data by 'Entity' and sum the total oil consumption for each country over the years
grouped_data = fuel_consumption.groupby('Entity')['Oil consumption(bbl)'].sum().reset_index()
# Rename the column for clarity
grouped_data.rename(columns={'Oil consumption(bbl)': 'Total Oil Consumption (barrels)'}, inplace=True)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#print top 10 rows (from 2 until 11, because row 1 is the total world consumption)
#print(sorted_data.iloc[1:12, :])

grouped_data.sort_values(by='Total Oil Consumption (barrels)', ascending=False, inplace=True)
#pick 6 countries based on oil consumption
top_countries = grouped_data['Entity'].head(6)
#make data base to store p value for each country
# DataFrame to store p-values and predictions
results_df = pd.DataFrame(columns=['Country', 'p-value'])

#plot data + predictions for each country on the same graph
for country in top_countries:
    #filter data according to country and skip USSR
    if country == 'Former U.S.S.R.':
        continue
    country_data = fuel_consumption[fuel_consumption['Entity'] == country]
    #prep and train the model
    X = country_data[['Year']]
    y = country_data['Oil consumption(bbl)']
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #create polynomial features (2nd degree worked the best excpet for US)
    if country == 'United States':
        poly = PolynomialFeatures(degree=15)
    elif country == 'Russia':
        poly = PolynomialFeatures(degree=2)
    else:
        poly = PolynomialFeatures(degree=2)
        
    X_train_poly = poly.fit_transform(X_train)
    X_valid_poly = poly.transform(X_valid)
    X_range = np.arange(1992, 2100, 1).reshape(-1, 1)
    
    #for the smaller plot
    X_range_small = np.arange(1992, 2019, 1).reshape(-1, 1)

    #create the model and fit data
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    #find accuracy of the model with r^2 and make calculate the predictions as well
    r_squared = model.score(X_valid_poly, y_valid)
    #make predictions on the validation data
    y_predicted = model.predict(poly.fit_transform(X_range))
    
    #smaller plot
    y_predicted_small = model.predict(poly.fit_transform(X_range_small))
    #create a new figure for the specific country
    plt.figure(figsize=(8, 5))

    #plot the main graph for the specific country
    # plt.subplot(121)
    # plt.scatter(X['Year'], y, label='Actual Data')
    # plt.plot(X_range, y_predicted, 'r-', label='Predictions')
    # plt.xlabel('Year')
    # plt.ylabel('Oil consumption (barrels)')
    # plt.title(f'{country} (Model Score = {r_squared:.3f})')
    # plt.grid(True)
    # plt.legend()

    #plot the smaller graph for the specific country
    # plt.subplot(122)
    plt.scatter(X['Year'], y, label='Actual Data')
    plt.plot(X_range_small, y_predicted_small, 'r-', label='Predictions')
    plt.xlabel('Year')
    plt.ylabel('Oil consumption (barrels)')
    plt.title(f'{country} (1992-2019)')
    plt.grid(True)
    plt.legend()

    #show the plots for the specific country
    plt.tight_layout()
    plt.savefig(f'plots\consumption\{country}_consumption_plot.png')
    # plt.show()

# ###################################### USED FOR REPORT #############################################

# The code below refers to the combining of plots

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures

# # Load data and only keep years between 1992 and 2019
# fuel_consumption = pd.read_csv("Fuel_production_vs_consumption.csv", encoding='utf-8')

# # Filter the data for the selected countries and years
# # ['India', 'Japan', 'Russia', 'China', 'US', 'World', 'Canada']
# countries = ['United States', 'China']
# fuel_consumption = fuel_consumption[(fuel_consumption['Year'] >= 1992) & (fuel_consumption['Year'] <= 2019) & fuel_consumption['Entity'].isin(countries)]

# # Convert 'Oil consumption(m)' from cubic meters to barrels (1 cubic meter = 6.28981 barrels) for the selected countries
# fuel_consumption['Oil consumption(bbl)'] = fuel_consumption['Oil consumption(m)'] * 6.28981

# # Group the data by 'Entity' (country) and sum the total oil consumption for each country over the years
# grouped_data = fuel_consumption.groupby('Entity')['Oil consumption(bbl)'].sum().reset_index()
# grouped_data.rename(columns={'Oil consumption(bbl)': 'Total Oil Consumption (barrels)'}, inplace=True)

# # Sort the data based on total oil consumption in descending order
# grouped_data.sort_values(by='Total Oil Consumption (barrels)', ascending=False, inplace=True)

# # Create a new figure for the selected countries
# plt.figure(figsize=(8, 5))

# # Prepare and train the model for each country
# for country in countries:
#     country_data = fuel_consumption[fuel_consumption['Entity'] == country]
#     X = country_data[['Year']]
#     y = country_data['Oil consumption(bbl)']
    
#     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
#     poly = PolynomialFeatures(degree=2)
#     X_train_poly = poly.fit_transform(X_train)
#     X_valid_poly = poly.transform(X_valid)
#     X_range_small = np.arange(1992, 2019, 1).reshape(-1, 1)
    
#     model = LinearRegression()
#     model.fit(X_train_poly, y_train)

#     r_squared = model.score(X_valid_poly, y_valid)
#     y_predicted_small = model.predict(poly.transform(X_range_small))

#     # Plot the actual data and predictions for each country
#     plt.scatter(X['Year'], y, label=f'{country} (Actual Data)')
#     plt.plot(X_range_small, y_predicted_small, label=f'{country} (Model Score = {r_squared:.3f})')

# plt.xlabel('Year')
# plt.ylabel('Oil consumption (barrels)')
# plt.title('Oil Consumption Trends (1992-2019) - World')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'plots\consumption\cosumptionCombined2_plot.png')
# plt.show()

