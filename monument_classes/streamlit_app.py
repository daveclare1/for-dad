import streamlit as st
import pandas as pd
import os

from scipy.cluster.hierarchy import fcluster
import numpy as np
import plotly.express as px

from dendrogramming import dendrogram_from_df
from map_tools import get_lat_long_from_os

dirname = os.path.dirname(__file__)
data_file = os.path.join(dirname, 'data_clean.csv')
map_file = os.path.join(dirname, 'gridrefs_clean.csv')

# Load the data
excludes_sites = ['Boskednan,']
excludes_features = ['other']

df = pd.read_csv(
    data_file,
    index_col=[1,0],
    )
df.index = df.index.droplevel(0)  # read it in with areas but then get rid of them for now
df = df.fillna(0)
df = df.replace(['y', 'Y', '?'], [1, 1, 0.5])
df = df.astype(int)
df.index = df.index.str.strip()
df = df[~df.index.isin(excludes_sites)]
df = df[df.columns[~df.columns.isin(excludes_features)]]


# divide and plot
dendrogram_plot, linkage = dendrogram_from_df(df, 'ward')


# analyse properties of groupings
n_clusters = 4
n_features = 10
df['clusters'] = fcluster(linkage, n_clusters, criterion='maxclust')
df.clusters = df.clusters.astype(str)  # for categorical plot labels
cluster_values = df.groupby('clusters').mean()

# compute difference metric
cluster_diffs = cluster_values.apply(lambda row: abs(cluster_values.iloc[cluster_values.index != row.name] - row).min(), axis=1)

# get top 10 columns by highest difference metric
col_order = cluster_diffs.max().T.sort_values(ascending=False).index
col_order = col_order[:n_features]

cluster_diffs = cluster_diffs[col_order]
cluster_values = cluster_values[col_order]

tick_pos = np.arange(1, n_clusters+1)
tick_labels = [str(t) for t in tick_pos]

cluster_value_plot = px.imshow(
    cluster_values, 
    color_continuous_scale='greys',
    )
cluster_value_plot.update_layout(
    title='Average value in cluster',
    yaxis = dict(
        tickmode = 'array',
        tickvals = tick_pos,
        ticktext = tick_labels
    )
    )

cluster_diff_plot = px.imshow(
    cluster_diffs,
    color_continuous_scale='greys',
    )
cluster_diff_plot.update_layout(
    title='Difference metric',
    yaxis = dict(
        tickmode = 'array',
        tickvals = tick_pos,
        ticktext = tick_labels
    ))


# describe the groupings
diff_mask = cluster_diffs > 0.1
cluster_values_masked = cluster_values.mask(~diff_mask, np.nan)

df_descriptions = pd.DataFrame(columns=['Cluster', 'Most Have', " Most Haven't"])

desc = ''
for i, row in cluster_values_masked.iterrows():
    desc = desc + f"{i}\n"
    has = ', '.join(row.index[row > 0.5])
    hasnt = ',  '.join(row.index[row < 0.5])
    df_descriptions.loc[i] = [i, has, hasnt]


# map
df_coords = pd.read_csv(
    map_file,
    index_col=[1,0],
    )
df_coords.index = df_coords.index.droplevel(0)
df_coords = df_coords.dropna()

latlong_data = df_coords.apply(lambda row: get_lat_long_from_os(row.Gridref), axis='columns', result_type='expand')
latlong_data.columns = ['latitude', 'longitude']
df_coords = pd.concat([df_coords, latlong_data], axis='columns')

df_complete = df_coords.join(df)
df_complete = df_complete.sort_values(by=['clusters','Name'])

map_plot = px.scatter_mapbox(
    df_complete,
    lat="latitude", 
    lon="longitude", 
    color='clusters',
    hover_name=df_complete.index,
    hover_data=['Gridref'],
    zoom=3, 
    height=600
    )

map_plot.update_layout(
    mapbox_style= "carto-positron",
    mapbox_zoom=4, 
    mapbox_center_lat=55,
    margin={"r":0,"t":0,"l":0,"b":0},
    )


# Layout
st.title("Monument Classification")
tabs = st.tabs([
    "Introduction",
    "Dendrogram",
    "Group Characteristics",
    "Map",
])

tabs[0].markdown("""
Agglomerative clustering is used to group together the most similar sites
step by step, each time adding another site to an existing group. The
hierarchy formed can then be subdivided by 'cutting the tree' at a level
where the cut make the groups most distinct from each other.

The process is presented in the tabs above.
""")
with tabs[0].expander("Complete Data"):
    st.dataframe(df_complete)
    csv = df_complete.to_csv().encode('utf-8')
    st.download_button(
        "Download",
        csv,
        "Site Data.csv",
        "text/csv",
    )

tabs[1].markdown("""
In the dendrogram below, longer horizontal lines means more difference between
the connected elements. 

For this grouping it was chosen to split at 4 clusters, as at this level the
difference between them is still quite large.

Note that the first three clusters are more similar to each other than to the
fourth.
""")
tabs[1].pyplot(dendrogram_plot)

tabs[2].markdown("""
For each identified cluster, the number of sites with each characteristic
is computed. From this information, the categories that best differentiate
clusters can be calculated. Below are two plots showing the ten categories
that exhibit the most difference across the clusters, from strongest 
differentiator on the left to 10th strongest on the right.

The upper plot shows how strongly each characteristic is present in each
cluster. A completely black square means every site in that cluster has that
characteristic, a completely white square means none of them do.
""")
tabs[2].plotly_chart(cluster_value_plot)
tabs[2].markdown("""
The second plot shows the most differentiating factors - totally black means
a characteristic unique to that category/cluster pairing, with shades of grey
showing ever weaker differentiators.
""")
tabs[2].plotly_chart(cluster_diff_plot)

tabs[2].markdown("""
Pulling out the darkest cells from the second plot as the strongest
differentiators, from the darkness of the first plot it can be observed whether 
the difference is due to a presence or an absence of a characteristic. This can
be summed up:
""")
# Inject CSS with Markdown
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
st.markdown(hide_table_row_index, unsafe_allow_html=True)
tabs[2].table(df_descriptions)

tabs[3].markdown("""
The sites can be plotted on a map, coloured by cluster. For convenience the
reference table of characteristics is repeated.
""")
tabs[3].table(df_descriptions)
tabs[3].plotly_chart(map_plot)
