import sqlite3
import os.path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import sklearn.decomposition
from plotnine import *

import ca


# Read data into pandas dataframe
def get_data() -> pd.DataFrame:
    print("Pulling data from database")

    connection = sqlite3.connect('data/twt-score.db')
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
    with open('data/legislators-current.yaml', 'r') as legislators_file:
        legislators = yaml.safe_load(legislators_file)
    connection = sqlite3.connect('data/twt-score.db')
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


def get_party_data() -> pd.DataFrame:
    connection = sqlite3.connect('data/twt-score.db')
    cursor = connection.cursor()
    members_response = cursor.execute(
        """
        SELECT member, bioguide, name, party FROM members
        """
    )
    members_result = members_response.fetchall()
    members = list(map(lambda x: x[0], members_result))
    bioguides = list(map(lambda x: x[1], members_result))
    names = list(map(lambda x: x[2], members_result))
    parties = list(map(lambda x: x[3], members_result))

    data = pd.DataFrame(
        {
            "member": pd.Series(members),
            "bioguide": pd.Series(bioguides),
            "name": pd.Series(names),
            "party": pd.Series(parties)
        }
    )

    return data


# Convert data to initial matrix (convert long to wide) with
# columns representing each user and rows representing each tweet
def initial_matrix(data: pd.DataFrame, drop=None, writeto=None) -> pd.DataFrame:
    print("Constructing initial matrix")

    data['values'] = 1
    mat = data.pivot_table(index='member', columns='user', values='values', aggfunc='sum', fill_value=0)
    mat = mat.astype(int)

    if drop is not None:
        mat = mat.drop(index=drop)

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


# Calculate ideology measure using principal component analysis
def PCA_measure(agr_mat: pd.DataFrame, writeto=None, varianceto=None) -> pd.DataFrame:
    print("Generating ideology scores using PCA")

    # Generate PCA
    pca = sklearn.decomposition.PCA()
    score = pca.fit_transform(agr_mat)

    n_columns = 3
    cols = [f"dim{i+1}" for i in range(n_columns)]

    score_df = pd.DataFrame(score[:, :n_columns], columns=cols, index=agr_mat.index)
    if writeto is not None:
        score_df.to_csv(writeto)

    variance_df = pd.DataFrame(pca.explained_variance_ratio_[:n_columns], index=cols, columns=['explained variance'])
    if varianceto is not None:
        variance_df.to_csv(varianceto)
    return score_df


# Calculate ideology measure using singular value decomposition
def SVD_measure(agr_mat: pd.DataFrame, writeto=None) -> pd.DataFrame:
    print("Generating ideology scores using SVD decomposition")

    # Generate SVD
    P = agr_mat.to_numpy()
    r = P.sum(axis=1)
    c = P.sum(axis=0).T
    # D_r_rsq = np.diag(1. / np.sqrt(r.A1))
    # D_c_rsq = np.diag(1. / np.sqrt(c.A1))
    U, D_a, V = np.linalg.svd(P, full_matrices=False)
    D_a = np.asmatrix(np.diag(D_a))
    V = V.T

    F = U * D_a
    G = V * D_a

    eigenvals = np.diag(D_a)**2

    cols = [f"dim{i+1}" for i in range(len(agr_mat.columns))]
    score_df = pd.DataFrame(F, columns=cols, index=agr_mat.index)
    return score_df

    # Save and return DataFrame
    scores_df = pd.DataFrame(U, columns=agr_mat.index)
    # if writeto is not None:
    #     scores_df.to_csv(writeto)
    return scores_df


def main() -> None:
    DROP = [817076257770835968, 2696643955, 339822881, 137823987, 29766367, 819744763020775425, 234469322,
            2853793517, 14845376, 3026622545, 1081350574589833221, 854715071116849157, 1344375287484723205,
            550401754, 2962868158]

    # if not os.path.exists('output/initial-matrix.csv'):
    #     twt_data = get_data()
    #     initial_mat = initial_matrix(twt_data, DROP, 'output/initial-matrix.csv')
    # if not os.path.exists('output/affiliation-matrix.csv'):
    #     initial_mat = pd.read_csv('output/initial-matrix.csv', index_col=0)
    #     affiliation_mat = affiliation_matrix(initial_mat, 'output/affiliation-matrix.csv')
    # if not os.path.exists('output/agreement-matrix.csv'):
    #     affiliation_mat = pd.read_csv('output/affiliation-matrix.csv', index_col=0)
    #     agreement_mat = agreement_matrix(affiliation_mat, 'output/agreement-matrix.csv')
    # else:
    #     initial_mat = pd.read_csv('output/initial-matrix.csv', index_col=0)
    #     affiliation_mat = pd.read_csv('output/affiliation-matrix.csv', index_col=0)
    #     agreement_mat = pd.read_csv('output/agreement-matrix.csv', index_col=0)

    twt_data = get_data()
    initial_mat = initial_matrix(twt_data, DROP, 'output/initial-matrix.csv')
    affiliation_mat = affiliation_matrix(initial_mat, 'output/affiliation-matrix.csv')

    # score = SVD_measure(agreement_mat)
    # score['dim1'] = 1 - score['dim1']
    # score_norm = (score-score.min())/(score.max()-score.min()) + 1
    # score_norm = np.logaddexp()

    CA = ca.CA(affiliation_mat)

    dim1 = CA.norm_G[:,0]
    dim1 = 1 - dim1 # Reverse values to match left-right ideological dimension
    dim2 = CA.norm_G[:,1]
    members = CA.rows
    score_df = pd.DataFrame({"member": members, "dim1": dim1, "dim2": dim2})
    party_data = get_party_data().set_index("member")
    score_df = score_df.join(party_data, "member")
    score_df.to_csv('output/score.csv', index=False)

    eigenvals = CA.eigenvals
    dims = np.arange(1, eigenvals.size + 1, 1)
    perc_eigen = 100. * eigenvals / eigenvals.sum()
    eigenval_data = pd.DataFrame({"Dimension": dims, "Eigenvalue": eigenvals, "Eigenvalue Percent": perc_eigen})
    eigenval_data.to_csv('output/eigenvalues.csv', index=False)

if __name__ == "__main__":
    main()