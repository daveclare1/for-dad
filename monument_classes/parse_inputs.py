# The raw xlsx has a few bugs and trip hazards, this should clean it up
import pandas as pd

EXCLUDES_SITES = ['Boskednan']
OTHER_RENAMES = [
    'other perimeter focal point',
    'other interior feature',
    'other outside feature',
]

DIAMETERS = {
    'diam 1': 5, 
    'diam 2': 15,
    'diam 3': 25,
    'diam 4': 40,
}

def parse_data(filename):
    df = pd.read_excel(filename, header=1)
    # Label the columns
    cols = df.columns.values
    cols[0] = "Name"
    others = 0
    for idx, c in enumerate(cols):
        if 'other.' in c:
            print(f"Replacing {c} with {OTHER_RENAMES[others]}")
            cols[idx] = OTHER_RENAMES[others]
            others = others + 1
    df.columns = cols
    # get rid of the area
    df.Name = df.Name.str.split(',').str[0]
    # trim whitespace
    df.Name = df.Name.str.strip()
    # Dispose of empty columns
    df = df.drop(columns=df.columns[df.columns.str.contains('Unnamed')])
    # Dispose of unwanted rows
    df = df[~df.index.isin(EXCLUDES_SITES)]
    # make sure everything is either y, ? or empty
    df = df.fillna('')
    df = df.replace(['>', '0y', 'Y', 'i'], ['?', 'y', 'y', 'y'])

    # # get diameters
    # df['diameter'] = df.apply(lambda row: DIAMETERS[row[-4:].str.len().idxmax()], axis=1)
    # # get rid of categorical diameters
    # df = df.drop(columns=df.columns[df.columns.str.contains('diam ')])

    return df


def parse_gridrefs(filename):
    df = pd.read_excel(filename, header=1)
    # print(df)
    df = df.drop(columns=df.columns[2:])
    # print(df)
    # Label the name columns
    df.columns = ['Name', 'Gridref']
    # get rid of the area
    df.Name = df.Name.str.split(',').str[0]
    # trim whitespace
    df.Name = df.Name.str.strip()
    df.Gridref = df.Gridref.str.strip()

    return df


if __name__ == '__main__':
    df = parse_data('data.xlsx')
    df.to_csv('data_clean.csv', index=False)

    df = parse_gridrefs('grid refs.xlsx')
    df.to_csv('gridrefs_clean.csv', index=False)