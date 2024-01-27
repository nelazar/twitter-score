import sqlite3
import os.path

import numpy as np
import pandas as pd


# Read data into pandas dataframe
def get_data() -> pd.DataFrame:
    print("Pulling data from database")

    connection = sqlite3.connect('twt-score.db')
    cursor = connection.cursor()

    response = cursor.execute(
        """
        SELECT DISTINCT user, tweet FROM users
        """
    )
    result = response.fetchall()

    connection.close()

    users = list(map(lambda x: x[0], result))
    tweets = list(map(lambda x: x[1], result))

    data = pd.DataFrame(
        {
            "users": pd.Series(users),
            "tweets": pd.Series(tweets)
        }
    )

    return data


# Convert data to initial matrix (convert long to wide) with
# columns representing each user and rows representing each tweet
def initial_matrix(data: pd.DataFrame, writeto=None) -> pd.DataFrame:
    print("Constructing initial matrix")

    data['values'] = 1
    mat = data.pivot_table(index='tweets', columns='users', values='values', fill_value=0)
    mat = mat.astype(int)

    if writeto is not None:
        mat.to_csv(writeto)

    return mat


# Convert matrix to affiliation matrix with diagonal entries representing the
# number of retweets on a given tweet and the off-diagonal entries representing
# the shared retweets between two tweets
def affiliation_matrix(mat: pd.DataFrame, writeto=None) -> pd.DataFrame:
    print("Constructing affiliation matrix")

    # Create empty matrix and set diagonal to retweet totals
    tweets = mat.index.to_numpy()
    count = len(tweets)
    aff_mat = np.empty([count, count], dtype=int)
    sums = mat.sum(axis=1).to_numpy()
    np.fill_diagonal(aff_mat, sums)

    # Fill lower triangle with shared retweet counts
    diag_count = 1
    for diag in range(1, count):
        index_count = 1
        for i in range(count - diag):
            row = diag + i
            col = i
            shared = 0
            for user in mat:
                if mat.loc[mat.index[tweets[row]], user] + mat.loc[mat.index[tweets[row]], user] == 2:
                    shared += 1
            aff_mat[row, col] = shared
            print(f"{index_count}/{count-diag}", end="\r")
            index_count += 1
        print(f"Completed diagonal {diag_count}/{count-1}")
        diag_count += 1
    
    aff_df = pd.DataFrame(aff_mat, columns=tweets, index=tweets)
    if writeto is not None:
        aff_df.to_csv(writeto)
    return aff_df


def main() -> None:
    if not os.path.exists('output/initial-matrix.csv'):
        twt_data = get_data()
        initial_mat = initial_matrix(twt_data, 'output/initial-matrix.csv')
    elif not os.path.exists('output/affiliation-matrix.csv'):
        initial_mat = pd.read_csv('output/initial-matrix.csv')
        affiliation_mat = affiliation_matrix(initial_mat, 'output/affiliation-matrix.csv')
        print(affiliation_mat)


if __name__ == "__main__":
    main()