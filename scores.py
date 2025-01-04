import sqlite3

import pandas as pd
import scipy.stats
from plotnine import *


# Add table to database
def add_table() -> None:

    connection = sqlite3.connect("data/twt-score.db")
    cursor = connection.cursor()

    # Create table with member information
    cursor.execute(
        """
        CREATE TABLE scores (
            member INTEGER PRIMARY KEY,
            bioguide TEXT,
            icpsr TEXT,
            name TEXT,
            party TEXT,
            twtscore1 REAL,
            twtscore2 REAL,
            nominate1 REAL,
            nominate2 REAL,
            cfscore REAL
        )
        """
    )


# Returns a dataframe of all of the different scores
def get_scores() -> pd.DataFrame:
    twt_scores = pd.read_csv('output/score.csv')
    twt_scores = twt_scores.convert_dtypes()
    nominate = pd.read_csv('data/HS118_members.csv', usecols=['icpsr', 'bioguide_id', 'nominate_dim1', 'nominate_dim2'])
    nominate = nominate.convert_dtypes().set_index("bioguide_id")
    cfscores = pd.read_csv('data/cfscores.csv', usecols=['election', 'recipient.cfscore', 'ICPSR2'])
    cfscores = cfscores[['recipient.cfscore', 'ICPSR2']]
    cfscores['ICPSR2'] = pd.to_numeric(cfscores['ICPSR2'], errors='coerce')
    cfscores = cfscores.dropna().convert_dtypes(convert_string=False).set_index("ICPSR2")

    scores = twt_scores.copy()
    scores = scores.join(nominate, "bioguide")
    scores = scores.join(cfscores, "icpsr")

    scores.rename(columns={"dim1": "twt_dim1", "dim2": "twt_dim2", "recipient.cfscore": "cfscore"}, inplace=True)

    return scores

# Test correlation between different variables
def test_correlations(scores, pairs):
    data = []
    for pair in pairs:
        score_cols = scores[[pair[0], pair[1]]].dropna()
        var1 = score_cols[pair[0]]
        var2 = score_cols[pair[1]]
        corr_test = scipy.stats.pearsonr(var1, var2)

        data.append({
            "var1": pair[0],
            "var2": pair[1],
            "r": corr_test.statistic,
            "p": corr_test.pvalue
        })

    return data

# Graph correlation
def graph_correlations(scores:pd.DataFrame, pairs):
    corr_data = test_correlations(scores, pairs)
    print(corr_data)
    party_codes = {"Republican": '0', "Independent": '1', "Democrat": '2'}
    party_col = pd.to_numeric(scores.replace({"party": party_codes})['party'])
    scores_np = scores.drop(columns=['bioguide', 'name'], inplace=False)
    scores_np['party'] = party_col
    scores_np = scores_np.astype(float)

    pair_labels = {
        "twt_dim1": "Twitter score 1st dimension",
        "twt_dim2": "Twitter score 2nd dimension",
        "nominate_dim1": "DW-NOMINATE 1st dimension",
        "nominate_dim2": "DW-NOMINATE 2nd dimension",
        "cfscore": "CFscore"
    }

    titles = {
        "twt_dim1_nominate_dim1": "Correlation between Twitter score and DW-NOMINATE\n(1st dimensions)",
        "twt_dim2_nominate_dim2": "Correlation between Twitter score and DW-NOMINATE\n(2nd dimensions)",
        "twt_dim1_cfscore": "Correlation between Twitter score (1st dimension)\nand CFscore"
    }

    labels = {
        "twt_dim1_nominate_dim1": {
            "x": [40],
            "y": [-0.65],
            "label": [f"R:           {corr_data[0]['r']:.4f}\np-value: {corr_data[0]['p']:.4f}"]
        },
        "twt_dim2_nominate_dim2": {
            "x": [20],
            "y": [-0.8],
            "label": [f"R:         {corr_data[1]['r']:.4f}\np-value: {corr_data[1]['p']:.4f}"]
        },
        "twt_dim1_cfscore": {
            "x": [40],
            "y": [-1],
            "label": [f"R:           {corr_data[2]['r']:.4f}\np-value: {corr_data[2]['p']:.4f}"]
        }
    }

    for pair in pairs:
        (
            ggplot() +
                geom_point(data=scores, mapping=aes(x=pair[0], y=pair[1], color='party')) +
                stat_smooth(data=scores_np, mapping=aes(x=pair[0], y=pair[1]), method="lm", se=False) +
                scale_color_manual({"Democrat": "#1874CD", "Independent": "#7CCD7C", "Republican": "#CD5555"}) +
                labs(x=pair_labels[pair[0]],
                     y=pair_labels[pair[1]],
                     title=titles[f"{pair[0]}_{pair[1]}"],
                     color="Party") +
                annotate("label",
                         x=labels[f"{pair[0]}_{pair[1]}"]['x'],
                         y=labels[f"{pair[0]}_{pair[1]}"]['y'],
                         label=labels[f"{pair[0]}_{pair[1]}"]['label'])
        ).save(f"visualizations/corr_{pair[0]}_{pair[1]}.png")

        (
            ggplot(scores_np, aes(x=pair[0], y=pair[1], group='party', color='factor(party)')) +
                geom_point() +
                stat_smooth(method="lm", se=False) +
                scale_color_manual({2: "#1874CD", 1: "#7CCD7C", 0: "#CD5555"},
                                   labels={0: 'Republican', 1: 'Independent', 2: 'Democrat'}) +
                labs(x=pair_labels[pair[0]],
                     y=pair_labels[pair[1]],
                     title=titles[f"{pair[0]}_{pair[1]}"],
                     color="Party")
        ).save(f"visualizations/corr_party_{pair[0]}_{pair[1]}.png")

# Graph different scores
def graph_scores(scores:pd.DataFrame):
    
    (
        ggplot(scores, aes(x="twt_dim1", y="twt_dim2", color="party")) +
            geom_point() +
            scale_color_manual({"Democrat": "#1874CD", "Independent": "#7CCD7C", "Republican": "#CD5555"}) +
            labs(x="Dimension 1",
                 y="Dimension 2",
                 title="First two dimensions of Twitter score",
                 color="Party")
    ).save("visualizations/twt_score.png")

    (
        ggplot(scores, aes(x="nominate_dim1", y="nominate_dim2", color="party")) +
            geom_point() +
            scale_color_manual({"Democrat": "#1874CD", "Independent": "#7CCD7C", "Republican": "#CD5555"}) +
            labs(x="Dimension 1",
                 y="Dimension 2",
                 title="DW-NOMINATE scores (118th Congress)",
                 color="Party")
    ).save("visualizations/nominate.png")

    (
        ggplot(scores, aes(x="cfscore", color="party", fill="party")) +
            geom_histogram(alpha=0.6, bins=30) +
            scale_color_manual({"Democrat": "#1874CD", "Independent": "#7CCD7C", "Republican": "#CD5555"}) +
            scale_fill_manual({"Democrat": "#1874CD", "Independent": "#7CCD7C", "Republican": "#CD5555"}) +
            labs(x="CFscore",
                 title="CFscore distribution by party (2022 federal elections)",
                 color="Party",
                 fill="Party")
    ).save("visualizations/cfscore.png")

# Graph eigenvalues
def graph_eigenvals():

    eigenvals = pd.read_csv("output/eigenvalues.csv")

    (
        ggplot(eigenvals, aes(x="Dimension", y="Eigenvalue Percent")) +
            geom_line() +
            labs(title="Eigenvalue percentages for each dimension of Twitter score",
                 subtitle="Indicates the percentage of the total variation each dimension explains")
    ).save("visualizations/eigenval-perc.png")


if __name__ == "__main__":
    scores = get_scores()

    # Test correlations
    pairs = [
        ("twt_dim1", "nominate_dim1"),
        ("twt_dim2", "nominate_dim2"),
        ("twt_dim1", "cfscore")
    ]
    graph_correlations(scores, pairs)
    graph_scores(scores)
    graph_eigenvals()
