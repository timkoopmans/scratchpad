import boto3
import pandas as pd
import random
import time

# Read the CSV file from https://www.kaggle.com/datasets/waqi786/songs-dataset-2000-2020-50k-records?resource=download
df = pd.read_csv('songs_2000_2020_50k.csv')

# Initialize the DynamoDB session
session = boto3.Session(profile_name='DevOpsAccessRole', region_name='us-east-1')
dynamodb = session.resource('dynamodb')

# Reference your table
table = dynamodb.Table('MusicCollection')

# Define the number of iterations for the load test
num_iterations = 10000

for _ in range(num_iterations):
    # Select a random row
    random_row = df.sample(n=1).iloc[0]

    # Extract artist, song title, and album title
    artist = random_row['Artist']
    song_title = random_row['Title']
    album_title = random_row['Album']

    # Define the item
    item = {
        'Artist': artist,
        'SongTitle': song_title,
        'AlbumTitle': album_title
    }

    # Put the item into the table
    table.put_item(Item=item)

    # Print a message for each iteration
    print(f"Inserted: {item}")

    # Optional: Add a small delay to avoid overwhelming the database
    time.sleep(0.1)

print("Load test completed.")