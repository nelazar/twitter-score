import os
import time
import datetime
import sqlite3

import yaml
import requests

from custom_types import Account


bearer_token = os.environ.get("BEARER_TOKEN")


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FollowersLookupPython"
    return r


# Makes a given request to Twitter's API
def connect_to_endpoint(url:str, params:dict) -> dict | list:
    response = requests.request("GET", url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code == 429:
        sleep_time = datetime.datetime.now() + datetime.timedelta(seconds=902)
        print(f"Pausing for rate limit until {sleep_time.strftime('%H:%M:%S')}")
        time.sleep(902)
        return connect_to_endpoint(url, params)
    elif response.status_code != 200:
        raise Exception(f"Request returned an error: {response.status_code} {response.text}")
    else:
        return response.json()
    

# Retrieve a list of Twitter accounts of congress members from the given files
def get_accounts(members:str, social_media:str) -> list[Account]:
    with (open(members, 'r') as legislators_file,
          open(social_media, 'r') as socials_file):
        
        legislators = yaml.safe_load(legislators_file)
        socials = yaml.safe_load(socials_file)
        
        accounts: list[Account] = []
        
        for legislator in legislators:
            id = legislator['id']['bioguide']
            for account in socials:
                if (account['id']['bioguide'] == id
                    and 'twitter_id' in account['social']):
                    if 'twitter' in account['social']:
                        handle = account['social']['twitter']
                    else:
                        handle = ''
                    accounts.append({
                        'id': id,
                        'name': legislator['name']['official_full'],
                        'handle': handle,
                        'twitter_id': account['social']['twitter_id']
                    })

    return accounts


# Returns a list of tweet IDs (limited to 10 to avoid tweet cap) given a user ID
def get_tweets(id:int, max_results:int=10) -> list[int]:
    url = f"https://api.twitter.com/2/users/{id}/tweets"
    params = {
        "tweet.fields": "id,created_at,referenced_tweets",
        "max_results": f"{max_results}"
    }

    tweets = []

    response = connect_to_endpoint(url, params)
    for tweet in response['data']:
        if 'referenced_tweets' not in tweet:
            tweets.append(int(tweet['id']))
        elif tweet['referenced_tweets'][0]['type'] == 'quoted':
            tweets.append(int(tweet['id']))

    return tweets


# Returns a list of user IDs that retweeted the given tweet
def get_retweets(tweet_id:int) -> list[int]:
    url = f"https://api.twitter.com/2/tweets/{tweet_id}/retweeted_by"

    users = []
    pagination_token = ''

    while True:

        if pagination_token == '':
            params = {
                "user.fields": "created_at,description",
                "max_results": "100"
            }
        else:
            params = {
                "user.fields": "created_at,description",
                "max_results": "100",
                "pagination_token": pagination_token
            }

        response = connect_to_endpoint(url, params)
        for user in response['data']:
            users.append(int(user['id']))

        if 'pagination_token' in response['meta']:
            pagination_token = response['meta'].keys()
        else:
            break

    return users


# Setup database
def setup_db() -> None:

    connection = sqlite3.connect("twt-score.db")
    cursor = connection.cursor()

    # Create table with member information
    cursor.execute(
        """
        CREATE TABLE members (
            member INTEGER PRIMARY KEY,
            bioguide TEXT,
            handle TEXT,
            name TEXT
        )
        """
    )

    # Create table with users, the tweet they retweeted, and the member that tweet was from
    cursor.execute(
        """
        CREATE TABLE users (
            user INTEGER NOT NULL,
            tweet INTEGER NOT NULL,
            member INTEGER NOT NULL,
            FOREIGN KEY (member) REFERENCES members (member)
        )
        """
    )


def main() -> None:

    setup = False
    pull_data = True

    if setup:
        setup_db()

    # Pull account data from yaml files
    if pull_data:
        print("Pulling account data from yaml files")
        accounts = get_accounts('legislators-current.yaml', 'legislators-social-media.yaml')

    # Setup connection to database
    print("Establishing connection to database")
    connection = sqlite3.connect("twt-score.db")
    cursor = connection.cursor()

    # Insert account data into database
    if setup:
        print("Inserting account data into database")
        cursor.executemany(
            """
            INSERT INTO members (member, bioguide, handle, name)
                VALUES (:twitter_id, :id, :handle, :name)
            """,
            accounts
        )
        connection.commit()

    # Iterate through all members and pull the related user accounts
    if pull_data:
        print("Reading tweets from Twitter/X API")
        total = len(accounts)
        count = 0
        print()
        for member in accounts:
            count += 1
            print(f"{count}/{total} - {member['name']}")
            tweets = get_tweets(member['twitter_id'])
            for tweet in tweets:
                retweets = get_retweets(tweet)
                for retweet in retweets:
                    cursor.execute(
                        """
                        INSERT INTO users (user, tweet, member) VALUES (?, ?, ?)
                        """,
                        (retweet, tweet, member['twitter_id'])
                    )
                    connection.commit()


if __name__ == '__main__':
    main()