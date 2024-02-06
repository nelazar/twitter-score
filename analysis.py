import sqlite3
import os.path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

import ca


# Read data into pandas dataframe
def get_data() -> pd.DataFrame:
    print("Pulling data from database")

    connection = sqlite3.connect('twt-score.db')
    cursor = connection.cursor()

    users_response = cursor.execute(
        """
        SELECT DISTINCT user, tweet, member FROM users
        """
    )
    users_result = users_response.fetchall()

    connection.close()

    users = list(map(lambda x: x[0], users_result))
    tweets = list(map(lambda x: x[1], users_result))
    members = list(map(lambda x: x[2], users_result))

    bioguides = []
    names = []
    parties = []
    with open('legislators-current.yaml', 'r') as legislators_file:
        legislators = yaml.safe_load(legislators_file)
    connection = sqlite3.connect('twt-score.db')
    cursor = connection.cursor()
    for member in members:
        members_response = cursor.execute(
            """
            SELECT bioguide, name, party FROM members WHERE member = ?
            """, (member,)
        )
        members_result = members_response.fetchone()
        bioguides.append(members_result[0])
        names.append(members_result[1])
        parties.append(members_result[2])

    data = pd.DataFrame(
        {
            "user": pd.Series(users),
            "tweet": pd.Series(tweets),
            "member": pd.Series(members),
            "bioguide": pd.Series(bioguides),
            "name": pd.Series(names),
            "party": pd.Series(parties)
        }
    )

    return data


# Convert data to initial matrix (convert long to wide) with
# columns representing each user and rows representing each tweet
def initial_matrix(data: pd.DataFrame, writeto=None) -> pd.DataFrame:
    print("Constructing initial matrix")

    data['values'] = 1
    mat = data.pivot_table(index='tweet', columns='user', values='values', fill_value=0)
    mat = mat.astype(int)

    if writeto is not None:
        mat.to_csv(writeto)

    return mat


# Convert matrix to affiliation matrix with diagonal entries representing the
# number of retweets on a given tweet and the off-diagonal entries representing
# the shared retweets between two tweets
def affiliation_matrix(mat: pd.DataFrame, writeto=None) -> pd.DataFrame:
    print("Constructing affiliation matrix")

    # Compute affiliation matrix
    # a_aff = XX' + D where X is the initial matrix and D is the degree matrix of X
    np_mat = mat.to_numpy()
    aff_mat = np_mat @ np_mat.T
    sums = mat.sum(axis=1).to_numpy()
    np.fill_diagonal(aff_mat, sums)
    
    # Save and return matrix
    aff_df = pd.DataFrame(aff_mat, columns=mat.index, index=mat.index)
    if writeto is not None:
        aff_df.to_csv(writeto)
    return aff_df


# Convert affiliation matrix to agreement matrix where each column represents the percentage of
# 'retweeters' of the given tweet who also retweeted each row's tweet
def agreement_matrix(aff_mat: pd.DataFrame, writeto=None) -> pd.DataFrame:
    print("Constructing agreement matrix")

    # Compute agreement matrix
    # G = a_aff/diag(a_aff)
    np_mat = aff_mat.to_numpy()
    agr_mat = (np_mat.T / np.diagonal(np_mat)).T

    # Save and return matrix
    agr_df = pd.DataFrame(agr_mat, columns=aff_mat.index, index=aff_mat.index)
    if writeto is not None:
        agr_df.to_csv(writeto)
    return agr_df


# Calculate ideology measure using singular value decomposition
def SVD_measure(agr_mat: pd.DataFrame, writeto=None) -> pd.DataFrame:
    print("Generating ideology scores using SVD decomposition")

    # Generate SVD
    np_mat = agr_mat.to_numpy()
    U, D_a, V = np.linalg.svd(np_mat, full_matrices=False)
    D_a = np.asmatrix(np.diag(D_a))
    V = V.T

    # Plot first two dimensions
    plt.figure(100)
    data = get_data()
    xmin, xmax = None, None
    ymin, ymax = None, None
    for i, tweet in enumerate(agr_mat.index):
        tweet_data = data.loc[data['tweet'] == tweet]
        x, y = U[i, 0], U[i, 1]
        if tweet_data.loc[tweet_data.index[0], 'party'] == "Democrat":
            party_color = "mediumblue"
        elif tweet_data.loc[tweet_data.index[0], 'party'] == "Republican":
            party_color = "indianred"
        else:
            party_color = "grey"
        plt.text(x, y, tweet_data.loc[tweet_data.index[0], 'name'], va='center', ha='center', color=party_color)
        xmin = min(x, xmin if xmin else x)
        xmax = max(x, xmax if xmax else x)
        ymin = min(y, ymin if ymin else y)
        ymax = max(y, ymax if ymax else y)

        if xmin and xmax:
            pad = (xmax - xmin) * 0.1
            plt.xlim(xmin - pad, xmax + pad)
        if ymin and ymax:
            pad = (ymax - ymin) * 0.1
            plt.ylim(ymin - pad, ymax + pad)
    
    plt.grid()
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.show()

    # Save and return DataFrame
    scores_df = pd.DataFrame(U, columns=agr_mat.index)
    # if writeto is not None:
    #     scores_df.to_csv(writeto)
    return scores_df


def main() -> None:
    if not os.path.exists('output/initial-matrix.csv'):
        twt_data = get_data()
        initial_mat = initial_matrix(twt_data, 'output/initial-matrix.csv')
    elif not os.path.exists('output/affiliation-matrix.csv'):
        initial_mat = pd.read_csv('output/initial-matrix.csv', index_col=0)
        affiliation_mat = affiliation_matrix(initial_mat, 'output/affiliation-matrix.csv')
    else: #if not os.path.exists('output/agreement-matrix.csv'):
        affiliation_mat = pd.read_csv('output/affiliation-matrix.csv', index_col=0)

        tweet_corr_analysis = ca.CA(affiliation_mat)

        plt.figure(100)
        tweet_corr_analysis.plot()

        plt.figure(101)
        tweet_corr_analysis.norm_plot()

        plt.show()
    #     agreement_mat = agreement_matrix(affiliation_mat, 'output/agreement-matrix.csv')
    # elif not os.path.exists('output/svd-scores.csv'):
    #     agreement_mat = pd.read_csv('output/agreement-matrix.csv', index_col=0)
    #     svd_scores = SVD_measure(agreement_mat, 'output/svd-scores.csv')


if __name__ == "__main__":
    main()