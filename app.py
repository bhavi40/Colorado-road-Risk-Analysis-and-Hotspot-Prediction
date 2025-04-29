import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import requests
from datetime import datetime
from geopy.geocoders import Nominatim
import geopy.distance
from sklearn.cluster import DBSCAN

app = Flask(__name__)


# Load the trained Random Forest model
with open('best_model_random_forest.pkl', 'rb') as file:
    model = pickle.load(file)

with open('hotspot_clusters.pkl', 'rb') as f:
    hotspot_clusters = pickle.load(f)

# Mapping for prediction labels
label_mapping = {0: 'slight', 1: 'serious', 2: 'fatal'}

# Define expected features for the Random Forest model
expected_features = [
    'Latitude', 'Longitude', 'Rush Hour', 'Alcohol Suspected',
    'System Code_City Street', 'System Code_County Road', 'System Code_Frontage Road',
    'System Code_Interstate Highway', 'System Code_State Highway',
    'Road Description_Alley Related', 'Road Description_At Intersection', 'Road Description_Auxiliary Lane',
    'Road Description_Crossover-Related ',  # Note the space at the end
    'Road Description_Driveway Access Related', 'Road Description_Express/Managed/HOV Lane',
    'Road Description_Intersection Related', 'Road Description_Mid-Block Crosswalk',
    'Road Description_Non-Intersection', 'Road Description_Parking Lot',
    'Road Description_Railroad Crossing Related', 'Road Description_Ramp',
    'Road Description_Ramp-related', 'Road Description_Roundabout', 'Road Description_Shared-Use Path or Trail',
    'Road Condition_Dry', 'Road Condition_Dry W/Visible Icy Road Treatment', 'Road Condition_Foreign Material',
    'Road Condition_Icy', 'Road Condition_Icy W/Visible Icy Road Treatment', 'Road Condition_Muddy',
    'Road Condition_Roto-Milled', 'Road Condition_Sand/Gravel', 'Road Condition_Slushy',
    'Road Condition_Slushy W/Visible Icy Road Treatment', 'Road Condition_Snowy',
    'Road Condition_Snowy W/Visible Icy Road Treatment', 'Road Condition_Wet',
    'Road Condition_Wet W/Visible Icy Road Treatment',
    'Lighting Conditions_Dark – Lighted', 'Lighting Conditions_Dark – Unlighted',
    'Lighting Conditions_Dawn or Dusk', 'Lighting Conditions_Daylight',
    'Weather Condition_Blowing Snow', 'Weather Condition_Clear', 'Weather Condition_Cloudy',
    'Weather Condition_Dust', 'Weather Condition_Fog', 'Weather Condition_Freezing Rain or Freezing Drizzle',
    'Weather Condition_Rain', 'Weather Condition_Sleet or Hail', 'Weather Condition_Snow',
    'Weather Condition_Wind',
    'Speed Limit Category_High', 'Speed Limit Category_Low', 'Speed Limit Category_Medium',
    'Speed Limit Category_Unknown'
]

# Define possible categories for System Code and Road Description
system_code_categories = ['City Street', 'County Road', 'Frontage Road', 'Interstate Highway', 'State Highway']
road_description_categories = [
    'Alley Related', 'At Intersection', 'Auxiliary Lane', 'Crossover-Related ',
    'Driveway Access Related', 'Express/Managed/HOV Lane', 'Intersection Related',
    'Mid-Block Crosswalk', 'Non-Intersection', 'Parking Lot', 'Railroad Crossing Related',
    'Ramp', 'Ramp-related', 'Roundabout', 'Shared-Use Path or Trail'
]


geolocator = Nominatim(user_agent="colorado_road_risk_analysis")

def get_coordinates(location):
    try:
        location_data = geolocator.geocode(location + ", Colorado, USA")
        if location_data:
            return location_data.latitude, location_data.longitude
        else:
            return None, None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None, None

def get_place_name(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), language='en')
        if location and location.address:
            # Extract a simplified place name (e.g., city or neighborhood)
            address_parts = location.address.split(', ')
            # Typically, the city or a significant place name is around the 2nd or 3rd part
            place_name = address_parts[1] if len(address_parts) > 1 else address_parts[0]
            return place_name
        else:
            return f"Lat: {lat:.4f}, Lon: {lon:.4f}"  # Fallback if reverse geocoding fails
    except Exception as e:
        print(f"Reverse geocoding error: {e}")
        return f"Lat: {lat:.4f}, Lon: {lon:.4f}"  # Fallback on error

# Simplified function to get route coordinates (fallback to straight line if API fails)
def get_route_coordinates(start_lat, start_lon, end_lat, end_lon):
    api_key = "5b3ce3597851110001cf6248933ad121d64d415f9c9a132f93c980aa"  # Replace with your OpenRouteService API key
    url = f"https://api.openrouteservice.org/v2/directions/driving-car?api_key={api_key}&start={start_lon},{start_lat}&end={end_lon},{end_lat}"
    try:
        response = requests.get(url)
        data = response.json()
        coordinates = data['features'][0]['geometry']['coordinates']  # List of [lon, lat]
        return [(lat, lon) for lon, lat in coordinates]
    except Exception as e:
        print(f"Error fetching route: {e}")
        # Fallback: Assume a straight line between start and end
        return [(start_lat, start_lon), (end_lat, end_lon)]

# Function to sample points along the route approximately every mile
def sample_route_points(coordinates):
    if not coordinates or len(coordinates) < 2:
        return coordinates

    # Calculate the total distance of the route
    total_distance = 0
    for i in range(len(coordinates) - 1):
        point1 = coordinates[i]
        point2 = coordinates[i + 1]
        distance = geopy.distance.distance(point1, point2).miles
        total_distance += distance

    # If the route is less than 1 mile, return just the start and end points
    if total_distance < 1:
        return [coordinates[0], coordinates[-1]]

    # Calculate cumulative distances along the route
    cumulative_distances = [0]
    for i in range(1, len(coordinates)):
        point1 = coordinates[i - 1]
        point2 = coordinates[i]
        distance = geopy.distance.distance(point1, point2).miles
        cumulative_distances.append(cumulative_distances[-1] + distance)

    # Sample points approximately every mile
    sampled_points = [coordinates[0]]  # Always include the start point
    target_distance = 3.0  # Start sampling at 3 mile
    for i in range(1, len(coordinates)):
        if cumulative_distances[i] >= target_distance:
            sampled_points.append(coordinates[i])
            target_distance += 1.0  # Move to the next mile marker

    # Always include the end point if it's not already included
    if sampled_points[-1] != coordinates[-1]:
        sampled_points.append(coordinates[-1])

    return sampled_points

# Function to get weather data using OpenWeatherMap API (with wind speed)
def get_weather(latitude, longitude):
    api_key = "eaa8a47accda83fbe8c11087f13f15e7"
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        weather = data['weather'][0]['main'].lower()
        wind_speed = data['wind']['speed']  # Wind speed in meters/second
        return weather, wind_speed
    except:
        return 'clear', 0  # Fallback to clear weather and no wind

# Function to determine if it's rush hour
def is_rush_hour(crash_hour, is_weekday):
    if is_weekday == 1:
        if (7 <= crash_hour <= 10) or (15 <= crash_hour <= 19):
            return 1
    return 0

# Function to infer lighting, weather, and road conditions
def infer_conditions(weather, wind_speed, crash_hour, system_code):
    if 6 <= crash_hour <= 18:
        lighting = 'Daylight'
    elif 5 <= crash_hour <= 6 or 18 <= crash_hour <= 19:
        lighting = 'Dawn or Dusk'
    else:
        if system_code in ['City Street', 'Interstate Highway']:
            lighting = 'Dark – Lighted'
        else:
            lighting = 'Dark – Unlighted'

    weather_conditions = {
        'clear': 'Clear',
        'clouds': 'Cloudy',
        'rain': 'Rain',
        'snow': 'Snow',
        'fog': 'Fog',
        'drizzle': 'Freezing Rain or Freezing Drizzle',
        'thunderstorm': 'Rain',
        'mist': 'Fog'
    }
    weather_condition = weather_conditions.get(weather, 'Clear')
    
    # Infer Blowing Snow if there's snow and high wind speed
    if weather == 'snow' and wind_speed > 10:  # Wind speed threshold (e.g., 10 m/s)
        weather_condition = 'Blowing Snow'

    road_conditions = {
        'clear': 'Dry',
        'clouds': 'Dry',
        'rain': 'Wet',
        'snow': 'Snowy',
        'fog': 'Wet',
        'drizzle': 'Wet',
        'thunderstorm': 'Wet',
        'mist': 'Wet'
    }
    road_condition = 'Snowy' if weather_condition == 'Blowing Snow' else road_conditions.get(weather, 'Dry')

    return lighting, weather_condition, road_condition

# Function to infer Speed Limit Category based on System Code
def infer_speed_limit_category(system_code):
    if system_code == 'Interstate Highway':
        return 'High'
    elif system_code in ['State Highway', 'City Street']:
        return 'Medium'
    else:
        return 'Low'

# Function to make prediction for a single point using the Random Forest model
def predict_for_point(lat, lon, alcohol_suspected, system_code, crash_hour, is_weekday):
    # Step 1: Generate dummy variables for System Code
    system_code_df = pd.DataFrame({'System Code': system_code_categories})
    system_code_dummies = pd.get_dummies(system_code_df, columns=['System Code'], dtype=float)

    # Step 2: Generate dummy variables for Road Description
    road_description_df = pd.DataFrame({'Road Description': road_description_categories})
    road_description_dummies = pd.get_dummies(road_description_df, columns=['Road Description'], dtype=float)

    # Step 3: Combine the dummy variables into a template DataFrame
    template_dummies = pd.concat([system_code_dummies, road_description_dummies], axis=1)

    # Step 4: Create the actual data for prediction
    data = {
        'Latitude': lat,
        'Longitude': lon,
        'Alcohol Suspected': alcohol_suspected,
        'System Code': system_code,
        'Road Description': 'Non-Intersection'
    }

    rush_hour = is_rush_hour(crash_hour, is_weekday)
    data['Rush Hour'] = rush_hour

    weather, wind_speed = get_weather(lat, lon)
    lighting, weather_condition, road_condition = infer_conditions(weather, wind_speed, crash_hour, system_code)

    df = pd.DataFrame([data])
    df = pd.get_dummies(df, columns=['System Code', 'Road Description'], dtype=float)

    # Step 5: Add numerical features to the template dummies
    for col in ['Latitude', 'Longitude', 'Rush Hour', 'Alcohol Suspected']:
        template_dummies[col] = 0.0

    # Step 6: Reindex the prediction DataFrame to match the template
    df = df.reindex(columns=template_dummies.columns, fill_value=0.0)

    # Step 7: Add the remaining features (Weather Condition, Road Condition, etc.)
    for feature in expected_features:
        if 'Weather Condition_' in feature and feature.split('_')[-1] == weather_condition:
            df[feature] = 1.0
        elif 'Lighting Conditions_' in feature and feature.split('_')[-1] == lighting:
            df[feature] = 1.0
        elif 'Road Condition_' in feature and feature.split('_')[-1] == road_condition:
            df[feature] = 1.0
        elif 'Speed Limit Category_' in feature and feature.split('_')[-1] == infer_speed_limit_category(system_code):
            df[feature] = 1.0
        elif feature not in df.columns:
            df[feature] = 0.0

    # Step 8: Final reindex to match expected_features
    df = df.reindex(columns=expected_features, fill_value=0.0)

    if len(df.columns) != len(expected_features):
        raise ValueError(f"Feature mismatch. Expected {len(expected_features)}, got {len(df.columns)}")

    prediction = model.predict(df)
    return prediction[0], weather_condition, rush_hour  # Return prediction, weather_condition, and rush_hour

# Function to filter data for hotspot prediction (reusing weather_condition and rush_hour)
def filter_data_hotspot(df, weather_condition, rush_hour):
    # Validate weather_condition against possible values
    valid_weather_conditions = ['Snow', 'Clear', 'Sleet Or Hail', 'Rain', 'Wind', 'Cloudy', 'Fog', 
                                'Blowing Snow', 'Freezing Rain Or Freezing Drizzle', 'Dust']
    # Standardize the input weather_condition to match the dataset
    weather_condition = weather_condition.title()
    if weather_condition not in valid_weather_conditions:
        return None, f"Invalid weather condition: {weather_condition}. Must be one of {valid_weather_conditions}"
    
    filtered_df = df[
        (df['Weather Condition'] == weather_condition) &
        (df['Rush Hour'] == rush_hour)
    ]
    
    if len(filtered_df) < 5:
        return None, f"Only {len(filtered_df)} crashes match the conditions. Need at least 5 crashes for DBSCAN."
    return filtered_df, None

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        source = request.form['source']
        destination = request.form['destination']
        alcohol_suspected = int(request.form['alcohol_suspected'])
        system_code = request.form['system_code']

        source_lat, source_lon = get_coordinates(source)
        dest_lat, dest_lon = get_coordinates(destination)

        if source_lat is None or dest_lat is None:
            return render_template('index.html', prediction=None, hotspots=None, error="Could not geocode source or destination. Please try again.")

        route_coords = get_route_coordinates(source_lat, source_lon, dest_lat, dest_lon)
        if not route_coords:
            return render_template('index.html', prediction=None, hotspots=None, error="Could not fetch route. Please try again.")

        sampled_points = sample_route_points(route_coords)
        if not sampled_points:
            return render_template('index.html', prediction=None, hotspots=None, error="No points sampled along the route.")

        current_time = datetime.now()
        crash_hour = current_time.hour
        is_weekday = 1 if current_time.weekday() < 5 else 0

        predictions = []
        weather_condition = None
        rush_hour = None
        for lat, lon in sampled_points:
            pred, wc, rh = predict_for_point(lat, lon, alcohol_suspected, system_code, crash_hour, is_weekday)
            predictions.append(pred)
            if weather_condition is None:
                weather_condition = wc
            if rush_hour is None:
                rush_hour = rh

        final_prediction = max(predictions)
        predicted_class = label_mapping[final_prediction]

        key = (weather_condition.title(), rush_hour)
        filtered_df_hotspot = hotspot_clusters.get(key)
        if filtered_df_hotspot is None:
            return render_template('index.html', prediction=predicted_class, hotspots=None, error="No hotspot data available for these conditions.")

        eps_miles = 0.5
        hotspots = []
        for lat, lon in sampled_points:
            place_name = get_place_name(lat, lon)
            for idx, crash in filtered_df_hotspot.iterrows():
                crash_lat = crash['Latitude']
                crash_lon = crash['Longitude']
                distance = geopy.distance.distance((lat, lon), (crash_lat, crash_lon)).miles
                if distance <= eps_miles:
                    hotspots.append(place_name)
                    break

        # Get top 3 unique hotspot place names
        unique_hotspots = list(dict.fromkeys(hotspots))[:3]

        return render_template('index.html', prediction=predicted_class, hotspots=unique_hotspots, source=source, destination=destination)

    return render_template('index.html', prediction=None, hotspots=None)

if __name__ == '__main__':
    app.run(debug=True)
