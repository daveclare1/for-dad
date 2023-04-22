import re

from OSGridConverter import grid2latlong
import plotly.express as px
import pandas as pd


def get_lat_long_from_os(os_ref):
    match = re.match(r"([A-Z]+)([0-9]+)", os_ref.upper())
    try:
        letters, digits = match.groups()
        precision = int(len(digits)/2)
        l = grid2latlong(f"{letters} {digits[:precision]} {digits[precision:]}")
        return l.latitude, l.longitude
    except:
        return None, None


if __name__ == '__main__':
    ref = get_lat_long_from_os("SH724713")
    print(ref.latitude, ref.longitude)

    df = pd.DataFrame({
                    'lat': [ref.latitude],
                    'lon': [ref.longitude],
                    })

    fig = px.scatter_mapbox(df, lat="lat", lon="lon", zoom=3, height=300)

    fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=4, mapbox_center_lat = 41,
        margin={"r":0,"t":0,"l":0,"b":0})

    fig.show()