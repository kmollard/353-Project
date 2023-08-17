# CMPT353Project
<h1>Analyzing Fossil Fuel Consumption, Gas Prices, and Electric Vehicle Sales for a Sustainable Future</h1>
<h5>Instructor: Greg Baker<br>
Team Fuel and EVs: Parmveer Dayal, Edoardo Rinaldi, Kyle Mollard</h5>

<h2>Intro</h2>
In this project, we aim to investigate these interrelated aspects to shed light on the potential implications for the energy sector and the environment.
More specifically, in our analysis we relied on data related to fossil fuel consumption to find the countries that consume the most and, subsequently, attempted to find whether there was a correlation between fuel consumption and electric vehicles sales.
Moreover, as we progressed with our analysis, we wa nted to explore the possibility of a correlation between gas prices and EV sales, to better understand whether the recent increase in gas prices has had an influence on consumers’ behavior toward electric cars.
<h2>Datasets and Python Files</h2>
<h4>Datasets (.csv) include:</h4>
1. Fuel_production_vs_consumption.csv for Fuel Consumption<br>
2. gas_prices.csv for gas prices<br>
3. IEA-EV-dataEV-salesCarsHistorical.csv for EV sales<br>
<h4>Python files (.py) include:</h4>
1. fuel.py for Fuel Consumption Analysis<br>
2. gas.py for Fuel Cost Analysis<br>
3. ev.py for EV Analysis<br>
4. correlation.ipynb for Fuel Consumption vs EV Sales correlation check<br>
<h2>Libraries</h2>
Required Libraries:<br>
numpy, pandas, matplotlib.pyplot, statsmodels.api, and sklearn<br><br>
Specific modules from sklearn include:<br>
model_selection.train_test_split<br>
linear_model.LinearRegression<br>
preprocessing.PolynomialFeatures<br>
neighbors.KNeighborsClassifier<br>
neighbors.KNeighborsRegressionn<br>
pipeline.make_pipeline<br>
metrics.r2_score<br>
metrics.mean_squared_error<br><br>
The following modules were explored, but not used in the final results:<br>
statsmodels.nonparametric.smoothers_lowess.lowess<br>
sklearn.ensemble.RandomForestRegressor<br>
sklearn.neural_network.MLPRegressor<br>
<h2>Commands and Order of Execution:</h2>
After cloning the repo, simply run the python files from the project directory to go through the data analysis of the report. These files were run with Python 3.9.12 and Python 3.11.1<br><br>
Example for fuel consumption:<br>
''..\project> python fuel.py''<br>
One exception to this is the correlations.ipynb to be run and explored in a notebook environment<br>
<h2>Expected Output Files:</h2>
Each separate python file produces the appropriate plots for each country.<br>
Plots are stored in their respective folders within the plots directory<br><br>
Example for fuel consumption:<br>
Each country (or the World) plot is stored in the plots/consumption folder<br>
<h3><i>Refer to the report for more references</h3>