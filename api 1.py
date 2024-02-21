import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# image and or visualisation get display here itself

from faker import Faker
import pandas as pd
import random
from sqlalchemy import create_engine, ForeignKey, Column, Integer, String, Float, DateTime, Boolean, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, Float, Boolean, Date
from sqlalchemy import ForeignKey, DateTime, func


from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.llms import Replicate
from langchain.chains import ConversationalRetrievalChain
import sqlite3
import os
from datetime import datetime

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory
from math import radians, sin, cos, sqrt, atan2

import random
import numpy as np
from langchain.chains import LLMChain, SequentialChain
import joblib
import re

from langchain_google_vertexai import VertexAI

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
app=FastAPI()
engine = create_engine('sqlite:///dummy_tabbi.db')

# Load the trucks DataFrame
trucks_df = pd.read_sql_table('trucks', engine)

# Choose features and target variable
numeric_features = ['Capacity', 'Latitude', 'Longitude']
categorical_features = ['FuelType']
target = 'AverageFuelConsumption'

X = trucks_df[numeric_features + categorical_features]
y = trucks_df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create transformers for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a linear regression model within a pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Function to predict average fuel consumption for a given truck ID
def predict_fuel_consumption(truck_id):
    truck_data = trucks_df[trucks_df['TruckID'] == truck_id][numeric_features + categorical_features]
    prediction = model.predict(truck_data)
    print('predict_fuel_consumption function has been read')
    return prediction[0]

# Example usage
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "lumen-b-ctl-047-e2aeb24b0ea0 4 (1).json"
llm = VertexAI(model_name="gemini-pro", temperature=0)

# random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Function to calculate Haversine distance between two points
def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    print('calculated haverson formula')

    return distance

prompt_template = """
You are asked to provide the details of the best truck and best driver for a trip from {source_location} to {destination_location} with a required capacity of {required_capacity} usually in tons.

Given an input question, you have seven tables: trucks, drivers, locations, fuel_stations, load_management, maintenance,
and review.
Trucks table is having information about truck.
Drivers table is having information about drivers.
For calculating distance, use haversine_distance formula.
Source Location latitude and longitude can be taken from locations table and for calculating distance use haversine_distance formula.
Destination Location latitude and longitude can be taken from locations table.

Extracting details from input :
"Create a function called extract_trip_details(input_string) to extract trip details.
Define a list of regular expression patterns for various input formats.
Iterate through patterns, using re.search() to find a match.
Extract source location, destination, and required capacity from match groups.
Return details if found, else None."

Instructions for calculating best_trucks :
Create a function for suggest_best_truck that takes four parameters: source_lat (latitude of the source location), source_lon (longitude of the source location), load_capacity (required load capacity), and conn (database connection).
Inside the function, query the trucks table from the provided database connection. Iterate over each truck fetched from the database, calculate the distance between the source location and the truck's location using the Haversine formula.
Check if the truck has sufficient capacity (truck['Capacity'] >= load_capacity) and is closer than the current best option. Return the details of the  best truck based on these criteria.

Instructions for calculating best_drivers :
create a function for suggest_best_driver that takes three parameters: truck_lat(latitude of the truck location), truck_lon(longitude of the truck location), and conn (database connection).
Inside the function, query the driver table from the provided database connection. Iterate over each driver fetched from the database, calculate the distance between the truck location and the driver's location using the Haversine formula.
Check which driver is closer to the truck location and having more experences. Return the details of the best drivers based on these criteria.

Possible scenarios include but are not limited to:
1. Retrieving trucks, drivers,  locations, fuel_stations, load_management, maintenance and review details.
2. Obtaining available drivers information.
3. Obtaining available truck information.
4. Obtaining average fuel consumption of a truck.
5. For best truck,use trucks table and take the truck which is available nearest to the source location and the capacity of the truck must be equal or more then the required capacity.
6. For best driver,use drivers table take the driver which is available nearest to the truck location and then having more Experience.
7. The total distnace for the trip is distance between the source location and the destination location.
8. The average fuel consumption is the AverageFuelConsumption of the truck multiply by the distance between source location and destination location.
9. The output should contains the following information with proper sequence and column name : truckid, driverid, total trip distance, average fuel consumption for the trip, truck capacity, truck fuel type
10. The output must be from dummy_tabbi.db only.

output format:
Donot provide the query, execute the query and only provide details that you got by executing the query
use this as given example and give the output in same format
dont provide the query, execute the query and provide the output of the query.

(ex:

'TruckID': id of truck,
'Capacity': capacity of truck,
'FuelType': 'fuel type of truck as petrol or diesel',
'AverageFuelConsumption': average fuel that truck is consuming,
'Truck Availability': the availability of truck,
'LastInspectionDate' : Truck last inspection date,
'Driver Availability': the availability of Driver,
'Distance': total distance from source to destination ,
'DriverID': id of driver,
'DriverName': 'name of driver',
'LicenseNumber': license number of the truck,
'PhoneNumber': 'phone number of driver',
'Experience': experence of driver in years,
'PreviousRating' : Previous Average rating of driver
'Estamated time':to get the estimated time divide the total distance by average speed of the truck and add 3 hours, then round off the value and show the value in total hours
)
Given the input question, suggest the best truck and best driver, along with their details.
"""

# #function to extract input details from the input string
def extract_trip_details(input_string):
    patterns = [
        r"from (\w+) to (\w+) for transporting (\d+) tons? of load",
        r"create a trip from (\w+) to (\w+) for (\d+) tons? of load",
        r"trip from (\w+) to (\w+) transporting (\d+) tons? of load",
    ]

    for pattern in patterns:
        match = re.search(pattern, input_string)
        if match:
            source_location = match.group(1)
            destination_location = match.group(2)
            required_capacity = int(match.group(3))
            return source_location, destination_location, required_capacity

    # If no pattern matches, return None for all values
    return None, None, None

# Create PromptTemplate
prompt = PromptTemplate(input_variables=["source_location", "destination_location", "required_capacity"], template=prompt_template)

# Create LLMChain
chain = LLMChain(llm=llm, output_key="trip_details", prompt=prompt)

# Define SequentialChain
seq_chain = SequentialChain(
    chains=[chain],
    input_variables=["source_location", "destination_location", "required_capacity"],
    output_variables=["trip_details"],
    verbose=True
)

# Connecting to the database


# Function to run the trip based on the input string
def run_trip(input_string, conn):
    # Extract trip details from the input string
    source_location, destination_location, required_capacity = extract_trip_details(input_string)
    
    # Check if any required information is missing
    if source_location is None:
        print("Please provide the source location.")
        return
    if destination_location is None:
        print("Please provide the destination location.")
        return
    if required_capacity is None:
        print("Please provide the required capacity.")
        return
    
    # Check if the cache exists
    cache_filename = "trip_cache.joblib"
    try:
        trip_cache = joblib.load(cache_filename)
    except FileNotFoundError:
        trip_cache = {}

    # Check if the trip details are already cached
    cache_key = (source_location, destination_location, required_capacity, input_string)
    if cache_key in trip_cache:
        trip_details = trip_cache[cache_key]
    else:
        # Run the SequentialChain with the extracted trip details
        trip_details = seq_chain.run(source_location=source_location, destination_location=destination_location, required_capacity=required_capacity, conn=conn)
        # Cache the trip details
        trip_cache[cache_key] = trip_details
        joblib.dump(trip_cache, cache_filename)

    return trip_details

# Example input string
class TripInput(BaseModel):
    source: str
    destination: str
    capacity: int


@app.get('/get_data')
async def fuel_conception():
    sample_truck_id = 15
    predicted_fuel_consumption = predict_fuel_consumption(sample_truck_id)
    print(f'Predicted Average Fuel Consumption for Truck {sample_truck_id}: {predicted_fuel_consumption}')
    return {'predicted_fuel_consumption': predicted_fuel_consumption}

@app.post('/trip_creation')
async def trip_api(trip_input: TripInput):
    conn = sqlite3.connect('dummy_tabbi.db')
    input_string = f"create a trip from {trip_input.source} to {trip_input.destination} for transporting {trip_input.capacity} tons of load"
    trip_details = run_trip(input_string, conn)
    return trip_details
#     else:
#         raise HTTPException(status_code=404, detail="Details not found")  # Raise HTTPException if details not found

# Run the trip
# trip_details = run_trip(input_string, conn)
# print(trip_details)
