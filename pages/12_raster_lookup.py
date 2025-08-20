"""
Streamlit demo of two England maps without shape files.
"""
# ----- Imports -----
import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import geopandas

# Custom functions:
import stroke_maps.load_data
import utilities_maps.maps as maps
import utilities_maps.plot_maps as plot_maps
import utilities_maps.container_inputs as inputs


# ###########################
# ##### START OF SCRIPT #####
# ###########################
# page_setup()
st.set_page_config(
    page_title='Raster lookup map demo',
    page_icon=':rainbow:',
    layout='wide'
    )

container_maps = st.empty()


# Import the full travel time matrix:
df_travel_times = pd.read_csv(
    './data_maps/lsoa_travel_time_matrix_calibrated.csv', index_col='LSOA')
# Rename index to 'lsoa':
df_travel_times.index.name = 'lsoa'


# #################################
# ########## USER INPUTS ##########
# #################################

# User inputs for which hospitals to pick:
df_units = stroke_maps.load_data.stroke_unit_region_lookup()

# Limit to England:
# df_units = df_units.loc[~df_units['icb'].isna()].copy()
# Sort by ISDN (approximates sort by region):
df_units = df_units.sort_values('isdn')

# Find where in the list the default options are.
# Ugly conversion to int from int64 so selectbox() can take it.
ind1 = int(np.where(df_units.index == 'LE15WW')[0][0])

# Select hospitals by name...
unit1_name = st.selectbox(
    'Hospital 1',
    options=df_units['stroke_team'],
    index=ind1
)

# ... then convert names to postcodes for easier lookup.
unit1 = df_units.loc[df_units['stroke_team'] == unit1_name].index.values[0]


# Colourmap selection
cmap_names = [
    'cosmic_r', 'viridis_r', 'inferno_r', 'neutral_r',
    ]
cmap_diff_names = [
    'iceburn_r', 'seaweed', 'fusion', 'waterlily'
    ]
with st.sidebar:
    st.markdown('### Colour schemes')
    cmap_name = inputs.select_colour_map(cmap_names, key='cmap_name', widget_label='Colour display')

    with st.form('Colour band setup'):
        v_min = st.number_input(
            'LHS vmin',
            min_value=0,
            max_value=480,
            step=5,
            value=0,
            )
        v_max = st.number_input(
            'LHS vmax',
            min_value=0,
            max_value=480,
            step=5,
            value=120,
            )
        step_size = st.number_input(
            'LHS step',
            min_value=5,
            max_value=60,
            step=5,
            value=30,
            )
        submitted = st.form_submit_button('Submit')


# Display names:
subplot_titles = [
    'Majority vote',
    'Scaled by area'
]
cmap_titles = ['Travel time (minutes)' for s in subplot_titles]


# #######################################
# ########## MAIN CALCULATIONS ##########
# #######################################
# # While the main calculations are happening, display a blank map.
# # Later, when the calculations are finished, replace with the actual map.
# with container_maps:
#     plot_maps.plotly_blank_maps(['', ''], n_blank=2)

# Pick out the data for hospital 1 only:
df_data = df_travel_times[[unit1]]

# Maximum values:
tmax = df_data.max().max()

# Load LSOA name to code lookup:
path_to_lsoa_lookup = os.path.join('data_maps', 'lsoa_fid_lookup.csv')
df_lsoa_lookup = pd.read_csv(path_to_lsoa_lookup)
# Merge LSOA codes into time data:
df_data = pd.merge(df_data, df_lsoa_lookup[['LSOA11NM', 'LSOA11CD']], left_on='lsoa', right_on='LSOA11NM', how='left').drop('LSOA11NM', axis='columns')


# ####################################
# ########## SETUP FOR MAPS ##########
# ####################################

# Load LSOA geometry:
path_to_raster = os.path.join('data_maps', 'rasterise_geojson_lsoa11cd_ew.csv')
path_to_raster_info = os.path.join('data_maps', 'rasterise_geojson_fid_ew_transform_dict.csv')
#
df_raster = pd.read_csv(path_to_raster)
transform_dict = pd.read_csv(path_to_raster_info, header=None).set_index(0)[1].to_dict()


extent = [
    transform_dict['xmin'],
    transform_dict['im_xmax'],
    transform_dict['im_ymin'],
    transform_dict['ymax'],
]
area_each_pixel = transform_dict['pixel_size']**2.0

# Manually remove Isles of Scilly:
df_raster = df_raster.loc[~(df_raster['LSOA11CD_majority'] == 'E01019077')]

# df_raster['xi'] = df_raster.index // int(transform_dict['height'])
# df_raster['yi'] = df_raster.index % int(transform_dict['height'])

# Calculate how much of each pixel contains land (as opposed to sea
# or other countries):
df_raster['total_area_covered'] = df_raster['area_total'] / area_each_pixel
# Pick out pixels that mostly contain no land:
mask_sea = (df_raster['total_area_covered'] <= (1.0 / 3.0))
# Remove the data here so that they're not shown.
df_raster = df_raster.loc[~mask_sea]

# ----- Majority vote data -----
# Bring in the data to be displayed:
df_raster = pd.merge(
    df_raster,
    df_data[['LSOA11CD', unit1]].rename(columns={unit1: 'majority'}),
    left_on='LSOA11CD_majority',
    right_on='LSOA11CD',
    how='left',
    suffixes=[None, '_data']
    )


# ----- Scaled data -----
def convert_df_to_2darray(df_raster, data_col, transform_dict):
    # Make a 1D array with all pixels, not just valid ones:
    raster_arr_maj = np.full(
        int(transform_dict['height'] * transform_dict['width']), np.NaN)
    # Update the values of valid pixels:
    raster_arr_maj[df_raster['i'].values] = df_raster[data_col].values
    # Reshape into rectangle:
    raster_arr_maj = raster_arr_maj.reshape(
        (int(transform_dict['width']), int(transform_dict['height']))).transpose()
    return raster_arr_maj


raster_arr_maj = convert_df_to_2darray(df_raster, 'majority', transform_dict)

calculate_scaled_arr = st.checkbox('Calculate scaled array')
if calculate_scaled_arr:
    # Make a new dataframe. First column contains main LSOA, second
    # column next LSOA, etc. for the max number of LSOAs in any pixel.
    # Pixels with fewer LSOA will contain mostly nan.

    # LSOA names:
    series_lsoa = df_raster['LSOA11CD']
    # Include escape characters \ before the square brackets:
    series_lsoa = series_lsoa.str.replace("\['", '')
    series_lsoa = series_lsoa.str.replace("'\]", '')
    # Convert into dataframe with one LSOA per column:
    df_lsoa_explode = pd.DataFrame(series_lsoa.str.split("', '").values.tolist())

    # Proportions:
    series_props = df_raster['area_prop']
    # Include escape characters \ before the square brackets:
    series_props = series_props.str.replace("\[", '')
    series_props = series_props.str.replace("\]", '')
    # Convert into dataframe with one LSOA per column:
    df_props_explode = pd.DataFrame(series_props.str.split(", ").values.tolist()).astype(float)

    # Convert LSOA names to data values.
    str_data = '!'.join(series_lsoa.values.flatten())
    for k, v in dict(zip(df_data['LSOA11CD'], df_data[unit1])).items():
        str_data = str_data.replace(k, str(v))
    series_data = pd.Series(str_data.split('!'))
    df_data_explode = pd.DataFrame(series_data.str.split("', '").values.tolist()).astype(float)
    # Build new dataframe with a single column for scaled data:
    df_raster_scale = (df_data_explode * df_props_explode).sum(axis='columns')
    df_raster_scale = pd.DataFrame(df_raster_scale).rename(columns={0: 'scaled'})
    # Merge in the pixel identifiers:
    df_raster_scale = pd.concat((df_raster_scale, df_raster['i']), axis='columns')

    # ----- Convert to 2D array -----
    raster_arr_sca = convert_df_to_2darray(df_raster_scale, 'scaled', transform_dict)
else:
    raster_arr_sca = raster_arr_maj
    subplot_titles[1] = subplot_titles[0]

burned_lhs = raster_arr_maj
burned_rhs = raster_arr_sca



# Load colour info:
cmap_lhs = inputs.make_colour_list(cmap_name)


# Load stroke unit coordinates:
gdf_unit_coords = stroke_maps.load_data.stroke_unit_coordinates()

# ----- Plotting -----
fig = make_subplots(
    rows=1, cols=2,
    horizontal_spacing=0.0,
    subplot_titles=subplot_titles
    )

fig.add_trace(go.Heatmap(
    z=burned_lhs,
    transpose=False,
    x0=transform_dict['xmin'],
    dx=transform_dict['pixel_size'],
    y0=transform_dict['im_ymin'],
    dy=transform_dict['pixel_size'],
    zmin=0,
    zmax=tmax,
    colorscale=cmap_lhs,
    colorbar=dict(
        thickness=20,
        # tickmode='array',
        # tickvals=tick_locs,
        # ticktext=tick_names,
        # ticklabelposition='outside top'
        title='Times (minutes)'
        ),
    name='times'
), row='all', col=1)

fig.add_trace(go.Heatmap(
    z=burned_rhs,
    transpose=False,
    x0=transform_dict['xmin'],
    dx=transform_dict['pixel_size'],
    y0=transform_dict['im_ymin'],
    dy=transform_dict['pixel_size'],
    zmin=0,
    zmax=tmax,
    colorscale=cmap_lhs,
    colorbar=dict(
        thickness=20,
        # tickmode='array',
        # tickvals=tick_locs,
        # ticktext=tick_names,
        # ticklabelposition='outside top'
        title='Times (minutes)'
        ),
    name='diff'
), row='all', col=2)

fig.update_traces(
    {'colorbar': {
        'orientation': 'h',
        'x': 0.0,
        'y': -0.2,
        'len': 0.5,
        'xanchor': 'left',
        'title_side': 'bottom'
        # 'xref': 'paper'
        }},
    selector={'name': 'times'}
    )
fig.update_traces(
    {'colorbar': {
        'orientation': 'h',
        'x': 1.0,
        'y': -0.2,
        'len': 0.5,
        'xanchor': 'right',
        'title_side': 'bottom'
        # 'xref': 'paper'
        }},
    selector={'name': 'diff'}
    )

# Stroke unit locations:
fig.add_trace(go.Scatter(
    x=gdf_unit_coords['BNG_E'],
    y=gdf_unit_coords['BNG_N'],
    mode='markers',
    name='units'
), row='all', col='all')

# Equivalent to pyplot set_aspect='equal':
fig.update_yaxes(col=1, scaleanchor='x', scaleratio=1)
fig.update_yaxes(col=2, scaleanchor='x2', scaleratio=1)

# Shared pan and zoom settings:
fig.update_xaxes(matches='x')
fig.update_yaxes(matches='y')

# Remove axis ticks:
fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

with container_maps:
    # Write to streamlit:
    st.plotly_chart(
        fig,
        use_container_width=True,
        # config=plotly_config
        )
