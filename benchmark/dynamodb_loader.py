import boto3
import pandas as pd
import random
import time
from concurrent.futures import ThreadPoolExecutor

# Read the CSV file
df = pd.read_csv('songs_2000_2020_50k.csv')

# Initialize the DynamoDB session
session = boto3.Session(profile_name='DevOpsAccessRole', region_name='us-east-1')
dynamodb = session.resource('dynamodb')

# Reference your table
table = dynamodb.Table('MusicCollection')

# Define the number of iterations for the load test
num_iterations = 10000
num_threads = 20

def insert_random_song():
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

# Use ThreadPoolExecutor to run the load test concurrently
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(insert_random_song) for _ in range(num_iterations)]

# Wait for all threads to complete
for future in futures:
    future.result()

print("Load test completed.")