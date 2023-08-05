import re
from math import floor

from OSGridConverter import grid2latlong
from pyproj import Transformer
import plotly.express as px
import pandas as pd

def irish_grid_letter_offsets(letter):
    letter_num = ord(letter) - 65
    # there is no I in the system
    if letter_num > 7:
        letter_num = letter_num - 1
    col = letter_num % 5
    row = floor(letter_num / 5)
    return (4e5 + col * 1e5, 9e5 - row * 1e5)


def convert_irish(letter, e, n):
    if len(str(e)) != len(str(n)):
        print(f"{e}, {n} - unequal grid measurements")
        raise ValueError("unequal grid measurements")
    # first convert from irish grid to ITM
    offset_e, offset_n = irish_grid_letter_offsets(letter)
    easting = offset_e + int(e) * 10**(5-len(str(e)))
    northing = offset_n + int(n) * 10**(5-len(str(n)))
    # then from ITM to lat long
    transformer = Transformer.from_crs("EPSG:2157", "EPSG:4326")
    return transformer.transform(easting, northing)


def convert_british(letters, e, n):
    l = grid2latlong(f"{letters} {e} {n}")
    return l.latitude, l.longitude


def get_lat_long_from_os(os_ref):
    match = re.match(r"([A-Z]+)\s*([0-9]+)", os_ref.upper())
    try:
        letters, digits = match.groups()
        precision = int(len(digits)/2)
        # print(letters, digits[:precision], digits[precision:])
        if len(letters) == 1:
            # print('Irish')
            return convert_irish(letters, digits[:precision], digits[precision:])
        else:
            # print('British')
            return convert_british(letters, digits[:precision], digits[precision:])

    except:
        print(f"FAIL: {os_ref}")
        return None, None


if __name__ == '__main__':
    print(convert_irish('J', 628, 495))
    # ref = get_lat_long_from_os("SH724713")
    # print(ref.latitude, ref.longitude)

    # df = pd.DataFrame({
    #                 'lat': [ref.latitude],
    #                 'lon': [ref.longitude],
    #                 })

    # fig = px.scatter_mapbox(df, lat="lat", lon="lon", zoom=3, height=300)

    # fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=4, mapbox_center_lat = 41,
    #     margin={"r":0,"t":0,"l":0,"b":0})

    # fig.show()