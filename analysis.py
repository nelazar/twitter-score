import sqlite3
import numpy as np
import pandas as pd


# Read data into pandas dataframe
def get_data() -> pd.DataFrame:
    connection = sqlite3.connect('twt-score.db')
    cursor = connection.cursor()

    response = cursor.execute(
        """
        SELECT DISTINCT (user, tweet) FROM users
        """
    )
    result = response.fetchall()

    connection.close()

    users = list(map(lambda x: x[0]), result)
    tweets = list(map(lambda x: x[1]), result)

    data = pd.DataFrame(
        {
            "users": pd.Series(users),
            "tweets": pd.Series(tweets)
        }
    )

    return data


# Convert data to initial matrix (convert long to wide) with
# columns representing each user and rows representing each tweet
def initial_matrix(data: pd.DataFrame) -> pd.DataFrame:
    data['values'] = 1
    mat = data.pivot_table(index='tweets', columns='users', values='values', fill_value=0)

    return mat


# Convert matrix to affiliation matrix with diagonal entries representing the
# number of retweets on a given tweet and the off-diagonal entries representing
# the shared retweets between two tweets
def initial_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    members = mat.index.to_numpy()
    aff_mat = np.empty(())


def main() -> None:
    twt_data = get_data()
    print(twt_data)


if __name__ == "__main__":
    main()