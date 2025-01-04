import numpy as np
import pandas as pd
from plotnine import *

score_df = pd.read_csv('output/score.csv')

(
    ggplot(score_df, aes(x="dim1", y="dim2", color="party")) +
        geom_point() +
        scale_color_manual({"Democrat": "#1874CD", "Independent": "#7CCD7C", "Republican": "#CD5555"}) +
        labs(title="First two dimensions of ")
).save("score.png")