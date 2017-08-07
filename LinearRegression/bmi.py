import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

bmi_data = pd.read_csv('./LinearRegression/bmi.csv')

# Train the model
model = LinearRegression()
model.fit(bmi_data[['BMI']], bmi_data[['Life expectancy']])

laos_life_exp = model.predict(21.07931)
print(laos_life_exp)