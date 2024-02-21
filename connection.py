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



fake = Faker('en_IN')  # Set the locale to India

# Create SQLite database engine
engine = create_engine('sqlite:///dummy_tabbi.db')

Base = declarative_base()

# Trucks Table with Availability
class Truck(Base):
    __tablename__ = 'trucks'
    TruckID = Column(Integer, primary_key=True)
    TruckNumber = Column(String)
    Capacity = Column(Integer)
    FuelType = Column(String)
    AverageFuelConsumption = Column(Float)
    Latitude = Column(Float)
    Longitude = Column(Float)
    Availability = Column(Boolean, default=True)
    LastMaintenanceDate = Column(Date)
    NextMaintenanceDate = Column(Date)
    InsuranceExpirationDate = Column(Date)
    RegistrationExpirationDate = Column(Date)
    AverageSpeed = Column(Float)
    

# Define Drivers Table with Availability
class Driver(Base):
    __tablename__ = 'drivers'
    DriverID = Column(Integer, primary_key=True)
    DriverName = Column(String)
    LicenseNumber = Column(Integer)
    PhoneNumber = Column(String)
    Experience = Column(Integer)
    Location = Column(String)
    Latitude = Column(Float)
    Longitude = Column(Float)
    Availability = Column(Boolean, default=True)
    PreviousRating = Column(Float, default=0.0)
    
# Define Locations Table
class Location(Base):
    __tablename__ = 'locations'
    LocationID = Column(Integer, primary_key=True)
    LocationName = Column(String)
    PinCode = Column(Integer)
    Latitude = Column(Float)
    Longitude = Column(Float)

# Define Fuel Stations Table
class FuelStation(Base):
    __tablename__ = 'fuel_stations'
    StationID = Column(Integer, primary_key=True)
    StationName = Column(String)
    Location = Column(String)
    PinCode = Column(Integer)
    FuelType = Column(String)
    FuelPrice = Column(Float)

# Define Load Management Table
class LoadManagement(Base):
    __tablename__ = 'load_management'
    LoadID = Column(Integer, primary_key=True)
    TruckID = Column(Integer, ForeignKey('trucks.TruckID'))
    DriverID = Column(Integer, ForeignKey('drivers.DriverID'))
    SourceLocation = Column(String)
    DestinationLocation = Column(String)
    LoadWeight = Column(Integer)
    DepartureTime = Column(DateTime, default=func.now())
    ArrivalTime = Column(DateTime, default=func.now())
    truck = relationship("Truck")
    driver = relationship("Driver")

# Define Maintenance Table with VehicleHealth
class Maintenance(Base):
    __tablename__ = 'maintenance'
    RecordID = Column(Integer, primary_key=True)
    TruckID = Column(Integer, ForeignKey('trucks.TruckID'))
    MaintenanceDate = Column(DateTime)
    Description = Column(String)
    Cost = Column(Integer)
    VehicleHealth = Column(String)
    LastInspectionDate = Column(Date)
    truck = relationship("Truck")

# Define Review Table
class Review(Base):
    __tablename__ = 'review'
    ReviewID = Column(Integer, primary_key=True)
    TruckID = Column(Integer, ForeignKey('trucks.TruckID'))
    DriverID = Column(Integer, ForeignKey('drivers.DriverID'))
    CustomerID = Column(String)
    DrivingSkills = Column(Float)
    Punctuality = Column(Float)
    CustomerService = Column(Float)
    VehicleMaintenance = Column(Float)
    CommunicationSkills = Column(Float)
    NavigationSkills = Column(Float)
    Adaptability = Column(Float)
    Professionalism = Column(Float)
    VehicleComfort = Column(Float)
    Rating = Column(Float)
    ReviewText = Column(String)
    Date = Column(DateTime)
    # Define relationships
    truck = relationship("Truck")
    driver = relationship("Driver")


# Create tables in the database
Base.metadata.create_all(bind=engine)

# Create DataFrames using Faker
trucks_df = pd.DataFrame([{
    'TruckID': i + 1,
    'TruckNumber': fake.random_int(1000, 9999),
    'Capacity': random.choice([8, 10, 12]),
    'FuelType': random.choice(['Petrol', 'Diesel']),
    'AverageFuelConsumption': round(random.uniform(3.0, 4.5), 2),
    'Latitude': float(fake.latitude()),
    'Longitude': float(fake.longitude()),
    'Availability': random.choice(['Available', 'Busy']),
    'LastMaintenanceDate': fake.date_between(start_date='-1y', end_date='today'),
    'NextMaintenanceDate': fake.date_between(start_date='-1y', end_date='today'),
    'InsuranceExpirationDate': fake.date_between(start_date='today', end_date='+1y'),
    'RegistrationExpirationDate': fake.date_between(start_date='today', end_date='+2y'),
    'AverageSpeed': random.choice([20, 21, 22,23, 24, 25, 26, 27])
} for i in range(90)])
drivers_df = pd.DataFrame([{
    'DriverID': i + 1,
    'DriverName': fake.name(),
    'LicenseNumber': fake.random_int(10000, 99999),
    'PhoneNumber': fake.phone_number(),
    'Experience': random.randint(1, 10),
    'Location': fake.city(),
    'Latitude': float(fake.latitude()),
    'Longitude': float(fake.longitude()),
    'Availability': random.choice(['Available', 'Busy']),
    'PreviousRating': round(random.uniform(3.0, 5.0), 2)
} for i in range(90)])

locations_df = pd.DataFrame([{
    'LocationID': i + 1,
    'LocationName': fake.city(),
    'PinCode': fake.random_int(100000, 999999),
    'Latitude': float(fake.latitude()),
    'Longitude': float(fake.longitude())
} for i in range(20)])  # Increase the range to add more locations

fuel_stations_df = pd.DataFrame([{
    'StationID': i + 1,
    'StationName': fake.company(),
    'Location': fake.city(),
    'PinCode': fake.random_int(100000, 999999),
    'FuelType': random.choice(['Petrol', 'Diesel']),
    'FuelPrice': round(random.uniform(80.0, 100.0), 2)
} for i in range(5)])

load_management_df = pd.DataFrame([{
    'LoadID': i + 1,
    'TruckID': fake.random_int(1, 100),
    'DriverID': fake.random_int(1, 90),
    'SourceLocation': fake.city(),
    'DestinationLocation': fake.city(),
    'LoadWeight': random.randint(5, 15),
    'DepartureTime': fake.date_time_between(start_date='-30d', end_date='now'),
    'ArrivalTime': fake.date_time_between(start_date='now', end_date='+30d')
} for i in range(10)])

maintenance_df = pd.DataFrame([{
    'RecordID': i + 1,
    'TruckID': fake.random_int(1, 100),
    'MaintenanceDate': fake.date_time_between(start_date='-60d', end_date='now'),
    'Description': fake.sentence(),
    'Cost': fake.random_int(500, 5000),
    'VehicleHealth': random.choice(['Excellent','Very Good','Good', 'Fair', 'Poor'])  # Add this line for VehicleHealth
} for i in range(10)])
review_df = pd.DataFrame([{
    'ReviewID': i + 1,
    'TruckID': fake.random_int(1, 100),
    'DriverID': fake.random_int(1, 90),
    'CustomerID': fake.uuid4(),
    'DrivingSkills': round(random.uniform(3.0, 5.0), 2),  # Example value for driving skills
    'Punctuality': round(random.uniform(3.0, 5.0), 2),  # Example value for punctuality
    'CustomerService': round(random.uniform(3.0, 5.0), 2),  # Example value for customer service
    'VehicleMaintenance': round(random.uniform(3.0, 5.0), 2),  # Example value for vehicle maintenance
    'CommunicationSkills': round(random.uniform(3.0, 5.0), 2),  # Example value for communication skills
    'NavigationSkills': round(random.uniform(3.0, 5.0), 2),  # Example value for navigation skills
    'Adaptability': round(random.uniform(3.0, 5.0), 2),  # Example value for adaptability
    'Professionalism': round(random.uniform(3.0, 5.0), 2),  # Example value for professionalism
    'VehicleComfort': round(random.uniform(3.0, 5.0), 2),  # Example value for vehicle comfort
    'ReviewText': fake.sentence(),
    'Date': fake.date_time_between(start_date='-365d', end_date='now')
} for i in range(10)])

# Define a function to calculate the driver rating based on review parameters
def calculate_driver_rating(driving_skills, punctuality, customer_service, vehicle_maintenance, communication_skills, navigation_skills, adaptability, professionalism, vehicle_comfort):
    # Define weights for each parameter (you may adjust these according to your preference)
    weights = {
        'driving_skills': 0.15,
        'punctuality': 0.1,
        'customer_service': 0.15,
        'vehicle_maintenance': 0.1,
        'communication_skills': 0.1,
        'navigation_skills': 0.1,
        'adaptability': 0.1,
        'professionalism': 0.1,
        'vehicle_comfort': 0.1
    }
    # Calculate the weighted average
    weighted_sum = (
        weights['driving_skills'] * driving_skills +
        weights['punctuality'] * punctuality +
        weights['customer_service'] * customer_service +
        weights['vehicle_maintenance'] * vehicle_maintenance +
        weights['communication_skills'] * communication_skills +
        weights['navigation_skills'] * navigation_skills +
        weights['adaptability'] * adaptability +
        weights['professionalism'] * professionalism +
        weights['vehicle_comfort'] * vehicle_comfort
    )
 
    # Normalize the weighted sum to get the rating in the range [0, 5]
    rating = max(0, min(5, weighted_sum / sum(weights.values())))
    return rating
 
# Calculate the rating based on driver performance parameters
review_df['Rating'] = review_df.apply(lambda row: calculate_driver_rating(
    row['DrivingSkills'], row['Punctuality'], row['CustomerService'],
    row['VehicleMaintenance'], row['CommunicationSkills'], row['NavigationSkills'],
    row['Adaptability'], row['Professionalism'], row['VehicleComfort']
), axis=1)
 

# Convert Decimal values to float
locations_df['Latitude'] = locations_df['Latitude'].astype(float)
locations_df['Longitude'] = locations_df['Longitude'].astype(float)

# Insert data into tables
trucks_df.to_sql('trucks', engine, index=False, if_exists='replace')
drivers_df.to_sql('drivers', engine, index=False, if_exists='replace')
locations_df.to_sql('locations', engine, index=False, if_exists='replace')
fuel_stations_df.to_sql('fuel_stations', engine, index=False, if_exists='replace')
load_management_df.to_sql('load_management', engine, index=False, if_exists='replace')
maintenance_df.to_sql('maintenance', engine, index=False, if_exists='replace')
review_df.to_sql('review', engine, index=False, if_exists='replace')

# Load tables into Pandas DataFrames
trucks_df_from_db = pd.read_sql_table('trucks', engine)
drivers_df_from_db = pd.read_sql_table('drivers', engine)
locations_df_from_db = pd.read_sql_table('locations', engine)
fuel_stations_df_from_db = pd.read_sql_table('fuel_stations', engine)
load_management_df_from_db = pd.read_sql_table('load_management', engine)
maintenance_df_from_db = pd.read_sql_table('maintenance', engine)
review_df_from_db = pd.read_sql_table('review', engine)

# Print the head of each DataFrame
print('all the tables created')
review_df_from_db = pd.read_sql_table('review', engine)
print(review_df_from_db.head())

