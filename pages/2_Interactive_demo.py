"""
Streamlit app template.

Because a long app quickly gets out of hand,
try to keep this document to mostly direct calls to streamlit to write
or display stuff. Use functions in other files to create and
organise the stuff to be shown. In this example, most of the work is
done in functions stored in files named container_(something).py

This folium code is directly copied from Elliot's HSMA exercise
https://github.com/hsma5/5b_geospatial_problems/blob/main/exercise/5B_Folium_map_Group-SOLUTION.ipynb
"""
# ----- Imports -----
import streamlit as st
from streamlit_folium import st_folium

# Importing libraries
import folium
from folium import plugins
from folium.features import GeoJsonPopup, GeoJsonTooltip
import pandas as pd
import json
import numpy as np
# import pickle
# import cPickle
# For importing colour maps:
import matplotlib.pyplot as plt
import leafmap.foliumap as leafmap
import matplotlib.cm

# For opening the cloud-optimised geotiff:
from rasterio import open as rast_open
# For the colour bar:
import branca

# Custom functions:
from utilities_maps.fixed_params import page_setup

from datetime import datetime


def import_geojson(group_hospital):
    # Choose which geojson to open:
    geojson_file = 'LSOA_' + group_hospital.replace(' ', '~') + '.geojson'
    # geojson_file = './old/LSOA_2011.geojson'
    # geojson_file = 'LSOA_(Dec_2011)_Boundaries_Super_Generalised_Clipped_(BSC)_EW_V3.geojson'

    # lsoa geojson
    # with open('./data_maps/lhb_scn_geojson/' + geojson_file) as f:
    with open('./data_maps/lhb_stp_geojson/' + geojson_file) as f:
        geojson_ew = json.load(f)

    ## Copy the LSOA11CD code to features.id within geojson
    for i in geojson_ew['features']:
        i['id'] = i['properties']['LSOA11NMW']
    return geojson_ew



def import_hospital_geojson(hospital_postcode, MT=False):
    # Choose which geojson to open:
    geojson_file = 'lsoa_nearest_'
    if MT is True:
        geojson_file += 'MT_'
    else:
        geojson_file += 'to_'
    geojson_file += hospital_postcode
    geojson_file += '.geojson'

    # geojson_file = './old/LSOA_2011.geojson'
    # geojson_file = 'LSOA_(Dec_2011)_Boundaries_Super_Generalised_Clipped_(BSC)_EW_V3.geojson'

    # lsoa geojson
    # with open('./data_maps/lhb_scn_geojson/' + geojson_file) as f:
    with open('./data_maps/lsoa_nearest_hospital/' + geojson_file) as f:
        geojson_ew = json.load(f)

    # ## Copy the LSOA11CD code to features.id within geojson
    # for i in geojson_ew['features']:
    #     i['id'] = i['properties']['LSOA11NMW']
    return geojson_ew


def import_cog(out_cog):
    with rast_open(out_cog, 'r') as ds:
        myarray_colours = ds.read()

    # For plotting, the array needs to be in the shape NxMx4. 
    # When imported using rasterio, the array is in the shape 4xNxM. 
    # `np.transpose()` reshapes the array to the required dimensions.
    myarray_colours = np.transpose(myarray_colours, axes=(1, 2, 0))
    return myarray_colours


def import_geojson(geojson_file=''):
    if len(geojson_file) < 1:
        geojson_file = 'LSOA_(Dec_2011)_Boundaries_Super_Generalised_Clipped_(BSC)_EW_V3_reduced3.geojson'
    with open('./data_maps/' + geojson_file) as f:
        geojson_ew = json.load(f)
    return geojson_ew


def draw_cog_on_map(clinic_map, file_name, layer_name='Cog layer', alpha=1.0, visible=True, featuregroup=None, which_map=0):

    cog_array = import_cog(file_name)
    # TO DO - pull out bounds from the input array 
    # (something in the notebook does this already)


    image = folium.raster_layers.ImageOverlay(
        name=layer_name,
        image=cog_array,#out_cog,#out_cog,
        bounds=[[49.8647411589999976, -6.4185476299999999], [55.8110685409999974, 1.7629415090000000]],
        # bounds=[[49.6739854059999999, -6.6230848580000004], [56.0018242949999987, 1.9674787370000004]],
        opacity=alpha,
        mercator_project=True,
        # colormap=colour_func,
        # interactive=True,
        # cross_origin=False,
        # zindex=1,
        overlay=False,
        show=visible
    )
    
    if which_map == 0:
        image.add_to(clinic_map)
    elif which_map == 1:
        image.add_to(clinic_map.m1)
    elif which_map == 2:
        image.add_to(clinic_map.m2)
    return image, clinic_map

    # if featuregroup is None:
    #     image.add_to(clinic_map)
    #     return image, clinic_map
    # else:
    #     featuregroup.add_child(image)
    #     return image, featuregroup
    # # return image, clinic_map


def draw_LSOA_outlines_on_map(clinic_map):
    geojson_ew = import_geojson()

    lsoa_outlines = folium.GeoJson(
        data=geojson_ew,
        popup=GeoJsonPopup(
            fields=['LSOA11NM'],
            aliases=[''],
            localize=True
            ),
        name='LSOA outlines',
        style_function=lambda y:{  
                'fillColor': 'rgba(0, 0, 0, 0)',
                'color': 'rgba(0, 0, 0, 0)',
                'weight':0.1,
            },
        highlight_function=lambda y:{'weight': 2.0, 'color': 'black'},  # highlight_function / hover_dict
        # smooth_factor=1.5,  # 2.0 is about the upper limit here
        # show=False if g > 0 else True
        )
    lsoa_outlines.add_to(clinic_map)
    return lsoa_outlines, clinic_map


def draw_catchment_IVT_on_map(clinic_map, fg=None):
    if fg is None:
        fg = folium.FeatureGroup(name='Nearest IVT hospitals', show=False)
    dir = './data_maps/lsoa_nearest_hospital/'
    import os
    files = os.listdir(dir)
    files = [file for file in files if '_MT_' not in file]
    for file in files:
        with open(dir + file) as f:
            geojson = json.load(f)
        fg.add_child(
            folium.GeoJson(
                data=geojson,
                # popup=GeoJsonPopup(
                #     fields=['LSOA11NM'],
                #     aliases=[''],
                #     localize=True
                #     ),
                # name='LSOA outlines',
                style_function=lambda y:{ 
                        'fillColor': 'rgba(0, 0, 0, 0)',
                        'color': 'rgba(255, 255, 255, 255)',
                        'weight':1.0,
                    },
                highlight_function=lambda y:{
                    'weight': 2.0,
                    'fillColor': 'rgba(255, 255, 255, 127)'
                    },  # highlight_function / hover_dict
                # smooth_factor=1.5,  # 2.0 is about the upper limit here
                # show=False
                )
        )
    fg.add_to(clinic_map)
    return fg, clinic_map


def draw_catchment_MT_on_map(clinic_map, fg=None):
    if fg is None:
        fg = folium.FeatureGroup(name='Nearest MT hospitals', show=False)
    dir = './data_maps/lsoa_nearest_hospital/'
    import os
    files = os.listdir(dir)
    files = [file for file in files if '_MT_' in file]
    for file in files:
        with open(dir + file) as f:
            geojson = json.load(f)
        fg.add_child(
            folium.GeoJson(
                data=geojson,
                # popup=GeoJsonPopup(
                #     fields=['LSOA11NM'],
                #     aliases=[''],
                #     localize=True
                #     ),
                # name='LSOA outlines',
                style_function=lambda y:{ 
                        'fillColor': 'rgba(0, 0, 0, 0)',
                        'color': 'rgba(127, 0, 0, 255)',
                        'weight':3.0,
                        # 'dashArray': '5, 5'
                    },
                highlight_function=lambda y:{
                    'weight': 5.0,
                    'fillColor': 'rgba(127, 0, 0, 127)'
                    },  # highlight_function / hover_dict
                # smooth_factor=1.5,  # 2.0 is about the upper limit here
                # show=False
                )
        )
    fg.add_to(clinic_map)
    return fg, clinic_map



def draw_map(lat_hospital, long_hospital, geojson_list, region_list, df_placeholder, df_hospitals, nearest_hospital_geojson_list, nearest_mt_hospital_geojson_list, choro_bins=6):
    # Create a map
    clinic_map = folium.Map(location=[lat_hospital, long_hospital],
                            zoom_start=9,
                            tiles='cartodbpositron',
                            # prefer_canvas=True,
                            # Override how much people can zoom in or out:
                            min_zoom=0,
                            max_zoom=18,
                            width=1200,
                            height=600
                            )

    # # Add markers
    # for (index, row) in df_clinics.iterrows():
    #     pop_up_text = f"The postcode for {row.loc['Name']} " + \
    #                     f"({row.loc['Clinic']}) is {row.loc['postcode']}"
    #     folium.Marker(location=[row.loc['lat'], row.loc['long']], 
    #                   popup=pop_up_text, 
    #                   tooltip=row.loc['Name']).add_to(clinic_map)
    # folium.features.GeoJsonTooltip(['travel_time_mins'])

    outcome_min = 0
    outcome_max = 1
    choro_bins = np.linspace(outcome_min, outcome_max, 7)

    # Make a new discrete colour map:
    # (colour names have to be #000000 strings or names)
    colormap = branca.colormap.StepColormap(
        vmin=outcome_min,
        vmax=outcome_max,
        colors=["red", "orange", "lightblue", "green", "darkgreen", 'blue'],
        caption='Placeholder',
        index=choro_bins
    )
    colormap.add_to(clinic_map)
    # fg.add_child(colormap)  # doesn't work


    # fg = folium.FeatureGroup(name="test")
    
    # Add choropleth
    for g, geojson_ew in enumerate(geojson_list):
        # a = folium.Choropleth(geo_data=geojson_ew,
        #                 name=region_list[g], # f'choropleth_{g}',
        #                 bins=choro_bins,
        #                 data=df_placeholder,
        #                 columns=['LSOA11NMW', 'Placeholder'],
        #                 key_on='feature.id',
        #                 fill_color='OrRd',
        #                 fill_opacity=0.5,
        #                 line_opacity=0.5,
        #                 legend_name='Placeholder',
        #                 highlight=True,
        #                 #   tooltip='travel_time_mins',
        #                 smooth_factor=1.5,
        #                 show=False if g > 0 else True
        #                 ).add_to(clinic_map)
        # st.write(g)
        # st.write(a)
        # st.write(' ')



        # # fg.add_child(
        folium.GeoJson(
            data=geojson_ew,
            tooltip=GeoJsonTooltip(
                fields=['LSOA11NMW'],
                aliases=[''],
                localize=True
                ),
            # lambda x:f"{x['properties']['LSOA11NMW']}",
            # popup=popup,
            name=region_list[g],
            style_function= lambda y:{
                # "fillColor": colormap(y["properties"]["Estimate_UN"]),
                "fillColor": colormap(df_placeholder[df_placeholder['LSOA11NMW'] == y['properties']['LSOA11NMW']]['Placeholder'].iloc[0]),
                # "fillColor": colormap(df_placeholder[df_placeholder['LSOA11NMW'] == y['geometries']['properties']['LSOA11NMW']]['Placeholder'].iloc[0]),
                'stroke':'false',
                'opacity': 0.5,
                'color':'black',  # line colour
                'weight':0.5,
                # 'dashArray': '5, 5'
            },
            highlight_function=lambda x: {'weight': 2.0},
            smooth_factor=1.5,  # 2.0 is about the upper limit here
            show=False if g > 0 else True
            ).add_to(clinic_map)
        # ))

        # folium.TopoJson(
        #     data=geojson_ew,
        #     # tooltip=GeoJsonTooltip(
        #     #     fields=['LSOA11NMW'],
        #     #     aliases=[''],
        #     #     localize=True
        #     #     ),
        #     # lambda x:f"{x['properties']['LSOA11NMW']}",
        #     # popup=popup,
        #     # name=region_list[g],
        #     object_path='LSOA_South~West',
        #     # style_function= lambda y:{
        #     #     # "fillColor": colormap(y["properties"]["Estimate_UN"]),
        #     #     # "fillColor": colormap(df_placeholder[df_placeholder['LSOA11NMW'] == y['properties']['LSOA11NMW']]['Placeholder'].iloc[0]),
        #     #     "fillColor": colormap(df_placeholder[df_placeholder['LSOA11NMW'] == y['geometries']['properties']['LSOA11NMW']]['Placeholder'].iloc[0]),
        #     #     'stroke':'false',
        #     #     'opacity': 0.5,
        #     #     'color':'black',  # line colour
        #     #     'weight':0.5,
        #     #     # 'dashArray': '5, 5'
        #     # },
        #     # highlight_function=lambda x: {'weight': 2.0},
        #     # smooth_factor=1.5,  # 2.0 is about the upper limit here
        #     # show=False if g > 0 else True
        #     ).add_to(clinic_map)

    # import matplotlib.colors
    # colours = list(matplotlib.colors.cnames.values())
    # # st.write(colours)
    # for g, geojson_ew in enumerate(nearest_hospital_geojson_list):
    #     colour_here = colours[g]


    #     # fg.add_child(
    #     folium.GeoJson(
    #         data=geojson_ew,
    #         # tooltip=GeoJsonTooltip(
    #         #     fields=['LSOA11NMW'],
    #         #     aliases=[''],
    #         #     localize=True
    #         #     ),
    #         # lambda x:f"{x['properties']['LSOA11NMW']}",
    #         # popup=popup,
    #         name='placeholder_'+str(g), #region_list[g],
    #         style_function= lambda y:{
    #             # "fillColor": , #f'{colour_here}',# 'rgba(0, 0, 0, 0)',
    #             'stroke':'false',
    #             'opacity': 0.5,
    #             'color':'black',  # line colour
    #             'weight': 3,
    #             'dashArray': '5, 5'
    #         },
    #         # highlight_function=lambda x: {'fillOpacity': 0.8},
    #         smooth_factor=1.5,  # 2.0 is about the upper limit here
    #         show=True, # if g > 0 else True
    #         overlay=False
    #     # ))
    #         ).add_to(clinic_map)


    # Problem - no way to set colour scale max and min,
    # get a new colour scale for each choropleth.


    # Add markers
    for (index, row) in df_hospitals.iterrows():
        # pop_up_text = f'{row.loc["Stroke Team"]}'

        if last_object_clicked_tooltip == row.loc['Stroke Team']:
            icon = folium.Icon(
                # color='black',
                # icon_color='white',
                icon='star', #'fa-solid fa-hospital',#'info-sign',
                # angle=0,
                prefix='glyphicon')#, prefix='fa')
        else:
            icon = None


        # fg.add_child(
        folium.Marker(location=[row.loc['lat'], row.loc['long']], 
                    # popup=pop_up_text, 
                    tooltip=row.loc['Stroke Team']
        # ))
                    ).add_to(clinic_map)
        # folium.Circle(
        #     radius=100, # metres
        #     location=[row.loc['lat'], row.loc['long']],
        #     popup=pop_up_text,
        #     color='black',
        #     tooltip=row.loc['Stroke Team'],
        #     fill=False,
        #     weight=1
        # ).add_to(clinic_map)
        # fg.add_child(
        #     folium.Marker(location=[row.loc['lat'], row.loc['long']], 
        #               popup=pop_up_text, 
        #               tooltip=row.loc['Name'],
        #               icon=icon
        #               )
        # )

    # # Add choropleth
    # fg.add_child(
    #     folium.Choropleth(geo_data=geojson_cornwall,
    #                     name='choropleth',
    #                     data=df_travel_times,
    #                     columns=['LSOA11CD', 'travel_time_mins'],
    #                     key_on='feature.id',
    #                     fill_color='OrRd',
    #                     fill_opacity=0.5,
    #                     line_opacity=0.5,
    #                     legend_name='travel_time_mins',
    #                     highlight=True
    #                     )
    #                     )#.add_to(clinic_map)


    # # This works for starting the map in this area
    # # with the max zoom possible.
    # folium.map.FitBounds(
    #     bounds=[(50, -4), (51, -3)],
    #     padding_top_left=None,
    #     padding_bottom_right=None,
    #     padding=None,
    #     max_zoom=None
    #     ).add_to(clinic_map)

    folium.map.LayerControl().add_to(clinic_map)
    # fg.add_child(folium.map.LayerControl())

    # Generate map
    # clinic_map
    output = st_folium(
        clinic_map,
        # feature_group_to_add=fg,
        returned_objects=[
            # 'bounds',
            # 'center',
            # "last_object_clicked", #_tooltip",
            # 'zoom'
        ],
        )
    st.write(output)
    # st.stop()


def draw_map_plotly(df_placeholder, geojson_ew, lat_hospital, long_hospital):
    import plotly.express as px

    fig = px.choropleth_mapbox(
        df_placeholder,
        geojson=geojson_ew, 
        locations='LSOA11NMW',
        color='Placeholder',
        color_continuous_scale="Viridis",
        # range_color=(0, 12),
        mapbox_style="carto-positron",
        zoom=9, 
        center = {"lat": lat_hospital, "lon": long_hospital},
        opacity=0.5,
        # labels={'unemp':'unemployment rate'}
    )
    # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    # fig.show()
    st.plotly_chart(fig)



def draw_map_circles(lat_hospital, long_hospital, df_placeholder, df_hospitals, df_lsoa_regions):
    # Create a map
    # import leafmap.foliumap as leafmap

    # clinic_map = leafmap.Map(
    clinic_map = folium.Map(
        location=[lat_hospital, long_hospital],
        zoom_start=9,
        tiles='cartodbpositron',
        # prefer_canvas=True,
        # Override how much people can zoom in or out:
        min_zoom=0,
        max_zoom=18,
        width=1200,
        height=600
        )

    outcome_min = 0
    outcome_max = 1
    choro_bins = np.linspace(outcome_min, outcome_max, 7)

    # Make a new discrete colour map:
    # (colour names have to be #000000 strings or names)
    colormap = branca.colormap.StepColormap(
        vmin=outcome_min,
        vmax=outcome_max,
        colors=["red", "orange", "lightblue", "green", "darkgreen", 'blue'],
        caption='Placeholder',
        index=choro_bins
    )
    colormap.add_to(clinic_map)
    # fg.add_child(colormap)  # doesn't work

    # Add markers
    for (index, row) in df_hospitals.iterrows():
        # pop_up_text = f'{row.loc["Stroke Team"]}'

        if last_object_clicked_tooltip == row.loc['Stroke Team']:
            icon = folium.Icon(
                # color='black',
                # icon_color='white',
                icon='star', #'fa-solid fa-hospital',#'info-sign',
                # angle=0,
                prefix='glyphicon')#, prefix='fa')
        else:
            icon = None


        # fg.add_child(
        folium.Marker(location=[row.loc['lat'], row.loc['long']], 
                    # popup=pop_up_text, 
                    tooltip=row.loc['Stroke Team']
        # ))
                    ).add_to(clinic_map)

    marker_list = []
    for i, row in df_lsoa_regions.iterrows():
        # # folium.Circle(
        # # folium.PolyLine(
        #     # radius=100, # metres
        #     locations=[[row.loc['LSOA11LAT'], row.loc['LSOA11LONG']]],
        #     # popup=pop_up_text,
        #     color='black',
        #     tooltip=row.loc['LSOA11NM'],
        #     fill=True,
        #     weight=1
        # ).add_to(clinic_map)
        # pass
        marker_list.append(
        folium.Marker(
            [row.loc['LSOA11LAT'], row.loc['LSOA11LONG']],
        ))#.add_to(clinic_map)
    
    marker_list.add_to(clinic_map)
    
    # folium.plugins.FastMarkerCluster(
    #     np.transpose([df_lsoa_regions['LSOA11LAT'], df_lsoa_regions['LSOA11LONG']]),
    # ).add_to(clinic_map)
        
    # folium.plugins.HeatMap(
    #     np.transpose([df_lsoa_regions['LSOA11LAT'], df_lsoa_regions['LSOA11LONG']]),
    # ).add_to(clinic_map)

    folium.map.LayerControl().add_to(clinic_map)

    # Generate map
    output = st_folium(
        clinic_map,
        returned_objects=[
        ],
        )
    st.write(output)



def draw_map_leafmap(lat_hospital, long_hospital, geojson_list, region_list, df_placeholder, df_hospitals, nearest_hospital_geojson_list, nearest_mt_hospital_geojson_list, choro_bins=6):
    
    import leafmap.foliumap as leafmap

    # Create a map
    clinic_map = leafmap.Map(location=[lat_hospital, long_hospital],
                            zoom_start=9,
                            tiles='cartodbpositron',
                            # prefer_canvas=True,
                            # Override how much people can zoom in or out:
                            min_zoom=0,
                            max_zoom=18,
                            width=1200,
                            height=600
                            )


    outcome_min = 0
    outcome_max = 1
    choro_bins = np.linspace(outcome_min, outcome_max, 7)

    # Make a new discrete colour map:
    # (colour names have to be #000000 strings or names)
    colormap = branca.colormap.StepColormap(
        vmin=outcome_min,
        vmax=outcome_max,
        colors=["red", "orange", "LimeGreen", "green", "darkgreen", 'blue'],
        caption='Placeholder',
        index=choro_bins
    )
    colormap.add_to(clinic_map)
    # clinic_map.add_colormap(colormap)
    
    # Add choropleth
    for g, geojson_ew in enumerate(geojson_list):

        # style = {#lambda y:{
        #         # "fillColor": colormap(y["properties"]["Estimate_UN"]),
        #         # "fillColor": colormap(df_placeholder[df_placeholder['LSOA11NMW'] == y['properties']['LSOA11NMW']]['Placeholder'].iloc[0]),
        #         # "fillColor": colormap(df_placeholder[df_placeholder['LSOA11NMW'] == y['geometries']['properties']['LSOA11NMW']]['Placeholder'].iloc[0]),
        #         'fillColor': 'red',
        #         'stroke':'false',
        #         'fillOpacity': 0.5,
        #         'color':'black',  # line colour
        #         'weight':0.5,
        #         # 'dashArray': '5, 5'
        #     }

        # st.write(geojson_ew)

        # def style(y):
        #     # st.write(y)
        #     style_list=[]
        #     for x in y['features']:
        #         style_dict = {#lambda y:{
        #             # "fillColor": colormap(y["properties"]["Estimate_UN"]),
        #             "fillColor": colormap(
        #                 df_placeholder[
        #                     df_placeholder['LSOA11NMW'] == x['properties']['LSOA11NMW']
        #                     ]['Placeholder'].iloc[0]
        #                 ),
        #             # "fillColor": colormap(df_placeholder[df_placeholder['LSOA11NMW'] == y['geometries']['properties']['LSOA11NMW']]['Placeholder'].iloc[0]),
        #             # 'fillColor': 'red',
        #             'stroke':'false',
        #             'fillOpacity': 0.5,
        #             'color':'black',  # line colour
        #             'weight':0.5,
        #             # 'dashArray': '5, 5'
        #         }
        #         style_list.append(style_dict)
        #     # st.write(style_list)
        #     return style_list

        # st.write(geojson_ew)
        # colour_list = [
        #     colormap(
        #                 df_placeholder[
        #                     df_placeholder['LSOA11NMW'] == x['properties']['LSOA11NMW']
        #                     ]['Placeholder'].iloc[0]
        #                 )
        #     for x in geojson_ew['features']
        #     ]

        # ValueError: highlight_function should be a function that accepts items from data['features'] and returns a dictionary.


        # # fg.add_child(
        folium.GeoJson(
        # clinic_map.add_geojson(
            # in_geojson=geojson_ew,
            data=geojson_ew,
            tooltip=GeoJsonTooltip(
                fields=['LSOA11CD'],
                aliases=[''],
                localize=True
                ),
            # # lambda x:f"{x['properties']['LSOA11NMW']}",
            # # popup=popup,
            # name=region_list[g],
            # style=style(geojson_ew),
            style_function=lambda y:{  # style / style_function
                    # "fillColor": colormap(y["properties"]["Estimate_UN"]),
                    "fillColor": colormap(
                        df_placeholder[
                            df_placeholder['LSOA11CD'] == y['properties']['LSOA11CD']
                            ]['Placeholder'].iloc[0]
                        ),
                    # "fillColor": colormap(df_placeholder[df_placeholder['LSOA11NMW'] == y['geometries']['properties']['LSOA11NMW']]['Placeholder'].iloc[0]),
                    # 'fillColor': 'red',
                    # 'fillColor': colour_list,
                    # 'stroke':'false',
                    # 'fillOpacity': 0.5,
                    # 'color':'black',  # line colour
                    'weight':0.1,
                    # 'dashArray': '5, 5'
                },
            highlight_function=lambda y:{'weight': 2.0},  # highlight_function / hover_dict
            # smooth_factor=1.5,  # 2.0 is about the upper limit here
            # show=False if g > 0 else True
            ).add_to(clinic_map)
        # ))
        # )

    # for g, geojson_ew in enumerate(geojson_list):
    #     clinic_map.add_data(
    #         data=geojson_ew
    #     )



    fg = folium.FeatureGroup(name='hospital_markers')
    # Add markers
    for (index, row) in df_hospitals.iterrows():
        # pop_up_text = f'{row.loc["Stroke Team"]}'

        # if last_object_clicked_tooltip == row.loc['Stroke Team']:
        #     icon = folium.Icon(
        #         # color='black',
        #         # icon_color='white',
        #         icon='star', #'fa-solid fa-hospital',#'info-sign',
        #         # angle=0,
        #         prefix='glyphicon')#, prefix='fa')
        # else:
        #     icon = None


        fg.add_child(
        folium.Marker(
        # clinic_map.add_marker(
            location=[row.loc['lat'], row.loc['long']],
                    # popup=pop_up_text,
                    tooltip=row.loc['Stroke Team']
        ))
                    # )#.add_to(clinic_map)

    fg.add_to(clinic_map)

    # folium.map.LayerControl().add_to(clinic_map)
    clinic_map.add_layer_control()
    # fg.add_child(folium.map.LayerControl())

    # Generate map
    hello = clinic_map.to_streamlit()
    # hello = clinic_map.show_in_browser()

    # with open('pickle_test.p', 'wb') as pickle_file:
    #     pickle.dump(hello, pickle_file)

    # st.write(type(hello))
    # st.write(hello)
    # clinic_map
    # output = st_folium(
    #     clinic_map,
    #     # feature_group_to_add=fg,
    #     returned_objects=[
    #         # 'bounds',
    #         # 'center',
    #         # "last_object_clicked", #_tooltip",
    #         # 'zoom'
    #     ],
    #     )
    # st.write(output)
    # st.stop()



from branca.element import MacroElement

from jinja2 import Template

class BindColormap(MacroElement):
    """Binds a colormap to a given layer.

    https://nbviewer.org/gist/BibMartin/f153aa957ddc5fadc64929abdee9ff2e

    Parameters
    ----------
    colormap : branca.colormap.ColorMap
        The colormap to bind.
    """
    def __init__(self, layer, colormap):
        super(BindColormap, self).__init__()
        self.layer = layer
        self.colormap = colormap
        # # For overlays layers:
        # self._template = Template(u"""
        # {% macro script(this, kwargs) %}
        #     {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
        #     {{this._parent.get_name()}}.on('overlayadd', function (eventLayer) {
        #         if (eventLayer.layer == {{this.layer.get_name()}}) {
        #             {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
        #         }});
        #     {{this._parent.get_name()}}.on('overlayremove', function (eventLayer) {
        #         if (eventLayer.layer == {{this.layer.get_name()}}) {
        #             {{this.colormap.get_name()}}.svg[0][0].style.display = 'none';
        #         }});
        # {% endmacro %}
        # """)  # noqa

        # For base layers:
        self._template = Template(u"""
        {% macro script(this, kwargs) %}
            {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
            {{this._parent.get_name()}}.on('layeradd', function (eventLayer) {
                if (eventLayer.layer == {{this.layer.get_name()}}) {
                    {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
                }});
            {{this._parent.get_name()}}.on('layerremove', function (eventLayer) {
                if (eventLayer.layer == {{this.layer.get_name()}}) {
                    {{this.colormap.get_name()}}.svg[0][0].style.display = 'none';
                }});
        {% endmacro %}
        """)  # noqa



def draw_map_tiff(df_hospitals, layer_name='Outcomes', alpha=0.6):


    # Create a map base without tiles:
    # outcome_map = leafmap.Map(
    outcome_map = folium.plugins.DualMap(
        location=[53, -2.5],  # Somewhere in the middle of the map
        zoom_start=6,
        # Override how much people can zoom in or out:
        # min_zoom=0,
        # max_zoom=18,
        # width=1200,
        # height=600,
        # Remove extra controls for stuff we don't need:
        draw_control=False,
        scale_control=False,
        search_control=False,
        measure_control=False,
        # control=False
        tiles=None,
        # layout='vertical'
        )

    # Now add the tiles and remove the option to toggle them on and off.
    # This means that no matter what layers we add later,
    # the tiles will never be removed.
    # https://stackoverflow.com/questions/61345801/
    # featuregroup-layer-control-in-folium-only-one-active-layer
    # New feature group:
    base_map = folium.FeatureGroup(
        name='Base map', overlay=True, control=False)
    # Place the tiles in the feature group:
    folium.TileLayer(tiles='cartodbpositron').add_to(base_map)
    # Draw on the map:
    base_map.add_to(outcome_map)


    # Draw the coloured background images.
    # Set one to be visible on load and the others to appear when
    # selected.
    tiff_file_strings = [
        'drip~ship~nlvo~ivt~added~utility',
        'mothership~nlvo~ivt~added~utility',
        'mothership~minus~dripship~nlvo~ivt~added~utility',
        'drip~ship~lvo~mt~added~utility',
        'mothership~lvo~mt~added~utility',
        'mothership~minus~dripship~lvo~mt~added~utility'
    ]

    cog_files = [
        f'data_maps/LSOA_{s}_cog.tif'
        for s in tiff_file_strings
    ]

    layer_names = [
        # nLVO
        'Drip and ship IVT nLVO added utility',
        'Mothership IVT/nLVO added utility',
        'nLVO advantage of Mothership (added utility)',
        # LVO
        'Drip and ship IVT/MT LVO added utility',
        'Mothership MT IVT/LVO added utility',
        'LVO advantage of Mothership (added utility)'
    ]

    tiff_layers = []
    for t, c_file in enumerate(cog_files):
        m = 1 if t < 3 else 2
        # Set whether this layer will be shown on startup:
        v = False if t > 0 else True
        tiff_layer, outcome_map = draw_cog_on_map(
            outcome_map,
            c_file,
            layer_name=layer_names[t],
            alpha=alpha,
            visible=v,
            which_map=m
            )
        tiff_layers.append(tiff_layer)



    outcome_min = 0.0261
    outcome_max = 0.1759
    outcome_cmap = 'inferno'
    outcome_cbar_label = 'Added utility'
    
    diff_min = -0.09620000000000009
    diff_max = 0.09620000000000009
    diff_cmap = 'bwr_r'
    diff_cbar_label = 'Advantage of Mothership (added utility)'

    colourmaps = []
    for m in [outcome_map.m1, outcome_map.m2]:
        # --- Outcome colourbar ---
        # Use these points as fixed colours with labels:
        choro_bins = np.linspace(outcome_min, outcome_max, 7)
        # Get colours as (R, G, B, A) arrays:
        colours = plt.get_cmap(outcome_cmap)(np.linspace(0, 1, len(choro_bins)))
        # Update alpha to match the opacity of the background image:
        colours[:, 3] = alpha
        # Convert colours to tuple so that branca understands them:
        colours = [tuple(colour) for colour in colours]
        
        # Drip and ship colour bar:
        colormap_dripship = branca.colormap.LinearColormap(
            vmin=outcome_min,
            vmax=outcome_max,
            colors=colours,
            caption=outcome_cbar_label,
            index=choro_bins
        )
        colourmaps.append(colormap_dripship)
        # outcome_map.add_child(colormap_dripship)
        # colormap_dripship.add_to(outcome_map.m2)


        # Mothership colour bar:
        colormap_mothership = branca.colormap.LinearColormap(
            vmin=outcome_min,
            vmax=outcome_max,
            colors=colours,
            caption=outcome_cbar_label,
            index=choro_bins
        )
        # outcome_map.add_child(colormap_mothership)
        colourmaps.append(colormap_mothership)


        # --- Difference colourbar ---
        # Use these points as fixed colours with labels:
        choro_bins = np.linspace(diff_min, diff_max, 7)
        # Get colours as (R, G, B, A) arrays:
        colours = plt.get_cmap(diff_cmap)(np.linspace(0, 1, len(choro_bins)))
        # Update alpha to match the opacity of the background image:
        colours[:, 3] = alpha
        # Convert colours to tuple so that branca understands them:
        colours = [tuple(colour) for colour in colours]

        # Make a new discrete colour map:
        colormap_diff = branca.colormap.LinearColormap(
            vmin=diff_min,
            vmax=diff_max,
            colors=colours,
            caption=diff_cbar_label,
            index=choro_bins
        )
        # outcome_map.add_child(colormap_diff)
        colourmaps.append(colormap_diff)


    # Bind colourmaps to each tiff image so that only one bar shows
    # up at once.
    for i in range(len(tiff_layers)):
        if i < 3:
            # m = outcome_map.m1 if i < 3 else outcome_map.m2
            # m.add_child(BindColormap(tiff_layers[i], colourmaps[i]))
            # BindColormap(tiff_layers[i], colourmaps[i]).add_to(m)
            # st.write(i, 'top')
            colourmaps[i].add_to(outcome_map.m1)
            BindColormap(tiff_layers[i], colourmaps[i]).add_to(outcome_map.m1)
        else:
            # pass

            colourmaps[i].add_to(outcome_map.m2)
            BindColormap(tiff_layers[i], colourmaps[i]).add_to(outcome_map.m2)
            # st.write(i, 'bottom')
            # colourmaps[i].add_to(outcome_map)
            # BindColormap(tiff_layers[i], colourmaps[i]).add_to(outcome_map)
    # outcome_map.add_child(BindColormap(tiff_layers[1], colormap_mothership))
    # outcome_map.add_child(BindColormap(tiff_layers[2], colormap_diff))


    # Nearest IVT hospitals:
    ivt_outlines, outcome_map = draw_catchment_IVT_on_map(outcome_map)

    # Nearest MT hospitals:
    mt_outlines, outcome_map = draw_catchment_MT_on_map(outcome_map)

    # # LSOA outlines:
    # lsoa_outlines, outcome_map = draw_LSOA_outlines_on_map(outcome_map)


    # Hospital markers:
    # Place all markers into a FeatureGroup so that
    # in the layer controls they can be shown or removed
    # with a single click, instead of toggling each marker
    # individually.
    fg_markers = folium.FeatureGroup(
        name='Hospital markers',
        # Remove from the layer control:
        control=False
        )
    # Iterate over the dataframe to get hospital coordinates
    # and to choose the colour of the marker.
    for (index, row) in df_hospitals.iterrows():
        if row.loc['Use_MT'] > 0:
            colour = 'red'
        elif row.loc['Use_IVT'] > 0:
            colour = 'white'
        else:
            colour = ''
        if len(colour) > 0:
            fg_markers.add_child(
                folium.CircleMarker(
                    radius=2.6, # pixels
                    location=[row.loc['lat'], row.loc['long']],
                    # popup=pop_up_text,
                    color='black',
                    tooltip=row.loc['Stroke Team'],
                    fill=True,
                    fillColor=colour,
                    fillOpacity=1,
                    weight=1,
                )
            )
    fg_markers.add_to(outcome_map)


    polygons = [
        # lsoa_outlines, 
        ivt_outlines, 
        mt_outlines,
        # # fg_nearest_hospitals
        ]

    # Put everything not later specified in this layer control:
    # outcome_map.add_layer_control(collapsed=False)
    folium.LayerControl(
        collapsed=False, 
        name='Background image'
        ).add_to(outcome_map)
    # Anything specified in further GroupedLayerControl boxes
    # will be removed from the previous control and appear in here:
    folium.plugins.GroupedLayerControl(
        {'Shapes': polygons},
        collapsed=False,
        exclusive_groups=False, # True for radio, false for checkbox
    ).add_to(outcome_map)

    # Set z-order of the elements:
    # (can add multiple things in here but the lag increases)
    outcome_map.keep_in_front(fg_markers)



    # Generate map
    # For single map:
    # outcome_map.to_streamlit()
    # For DualMap:
    # st.components.v1.html(outcome_map._repr_html_(), height=1200, width=800)
    # st.write(outcome_map)
    # outcome_map.show()
    outcome_map.save('html_dual_test.html')



# ###########################
# ##### START OF SCRIPT #####
# ###########################
page_setup()

# Title:
st.markdown('# Folium map test')


startTime = datetime.now()


path_to_html = './html_dual_test.html' 
# path_to_html = 'https://github.com/samuel-book/streamlit_map_lsoa_outcomes/blob/main/html_test.html'

with open(path_to_html, 'r') as f:
    html_data = f.read()


time1 = datetime.now()
st.write('Time to load HTML:', time1 - startTime)

# raw_html = ('''
# <html>
# <head>
# </head>
# <body>
#   <p>
#   <div style = "text-align: left;">
#     <embed style="border: none;" src="./html_test.html" dpi="300" width="100%" height="600px" />
#   </div>
#   </p>
# </body>
# </html>
# '''
# )
# st.markdown(html_data, unsafe_allow_html=True)
st.components.v1.html(html_data, height=600)
# st.components.v1.iframe('4_Project', height=600)

time2 = datetime.now()
st.write('Time to draw map:', time2 - time1)


# with open(path_to_html, 'r') as f:
#     html_data = f.read()

# time3 = datetime.now()
# st.write('Time to read HTML:', time3 - time2)

# st.components.v1.html(html_data, height=600)


# time4 = datetime.now()
# st.write('Time to draw map:', time4 - time3)

st.write('done')
st.stop()



## Load data files
# Hospital info
df_hospitals = pd.read_csv("./data_maps/stroke_hospitals_22_reduced.csv")

# # Select a hospital of interest
# hospital_input = st.selectbox(
#     'Pick a hospital',
#     df_hospitals['Stroke Team']
# )

# # Find which region this hospital is in:
# df_hospitals_regions = pd.read_csv("./data_maps/hospitals_and_lsoas.csv")
# df_hospital_regions = df_hospitals_regions[df_hospitals_regions['Stroke Team'] == hospital_input]

# long_hospital = df_hospital_regions['long'].iloc[0]
# lat_hospital = df_hospital_regions['lat'].iloc[0]
# region_hospital = df_hospital_regions['RGN11NM'].iloc[0]
# if region_hospital == 'Wales':
#     group_hospital = df_hospital_regions['LHB20NM'].iloc[0]
# else:
#     group_hospital = df_hospital_regions['STP19NM'].iloc[0]

# # All hospital postcodes:
# hospital_postcode_list = df_hospitals_regions['Postcode']

# travel times
# df_travel_times = pd.read_csv("./data_maps/clinic_travel_times.csv")
df_travel_matrix = pd.read_csv('./data_maps/lsoa_travel_time_matrix_calibrated.csv')
LSOA_names = df_travel_matrix['LSOA']
# st.write(len(LSOA_names), LSOA_names[:10])
placeholder = np.random.rand(len(LSOA_names))
table_placeholder = np.stack([LSOA_names, placeholder], axis=-1)
# st.write(table_placeholder)
df_placeholder = pd.DataFrame(
    data=table_placeholder,
    columns=['LSOA11NMW', 'Placeholder']
)
# st.write(df_placeholder)

# LSOA lat/long:

df_lsoa_regions = pd.read_csv('./data_maps/LSOA_regions.csv')




# # geojson_ew = import_geojson(group_hospital)

# region_list = [
#     'Devon',
#     'Dorset',
#     'Cornwall and the Isles of Scilly',
#     'Somerset'
# ]
# region_list = ['South West']

geojson_list = []
# nearest_hospital_geojson_list = []
# nearest_mt_hospital_geojson_list = []
# for region in region_list:
#     geojson_ew = import_geojson(region)
#     geojson_list.append(geojson_ew)

# # geojson_file = 'LSOA_South~West_t.geojson'


# # Make a list of the order of LSOA codes in the geojson that made the geotiff
# LSOA_names = []
# for i in geojson_ew['features']:
#     LSOA_names.append(i['properties']['LSOA11CD'])
# # st.write(LSOA_names)

# st.write(LSOA_names[])

# # st.write(len(LSOA_names), LSOA_names[:10])
# # LSOA_names = df_travel_matrix['LSOA']
# placeholder = np.random.rand(len(LSOA_names))
# table_placeholder = np.stack(np.array([LSOA_names, placeholder], dtype=object), axis=-1)
# # st.write(table_placeholder)
# df_placeholder = pd.DataFrame(
#     data=table_placeholder,
#     columns=['LSOA11CD', 'Placeholder']
# )

# st.write(df_placeholder)


# st.write(geojson_ew)

# # Copy the LSOA11CD code to features.id within geojson
# for i in geojson_ew['features']:
#     i['id'] = i['properties']['LSOA11NMW']

# for i in geojson_ew['objects']['LSOA_South~West']['geometries']:
#     i['id'] = i['properties']['LSOA11NMW']
# geojson_list = [geojson_ew]

# for hospital_postcode in hospital_postcode_list:
#     try:
#         nearest_hospital_geojson = import_hospital_geojson(hospital_postcode)
#         nearest_hospital_geojson_list.append(nearest_hospital_geojson)
#     except FileNotFoundError:
#         pass

#     try:
#         nearest_mt_hospital_geojson = import_hospital_geojson(hospital_postcode, MT=True)
#         nearest_mt_hospital_geojson_list.append(nearest_mt_hospital_geojson)
#     except FileNotFoundError:
#         pass

# st.write(geojson_ew['features'][0])


time4 = datetime.now()



with st.spinner(text='Drawing map'):
    draw_map_tiff(
        # myarray_colours,
        # geojson_list,
        df_hospitals,
        # layer_name=outcome_column
        )


time5 = datetime.now()

st.write('Time to draw map:', time5 - time4)


# # ----- The end! -----
