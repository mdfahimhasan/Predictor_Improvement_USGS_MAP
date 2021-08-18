import itertools
import os
from glob import glob
import geopandas as gpd
import pandas as pd
from math import sqrt
from sklearn.neighbors import KDTree
import numpy as np
from pyproj import Transformer
from Python_Files.maplibs.rasterops import read_raster_as_arr, create_cdl_raster_aiwum1

field_19 = r'..\Files_From_Emmy\2019_unique_field_polygons\2019_unique_field_polygons.shp'
fields_all = r'..\Files_From_Emmy\Permitted_boundaries_2018_2019\Permit_Boundaries2019.shp'
vmp = r'.\well_field_connection\VMP_readings_2014_2020.csv'


# comparing pump crop data with 0.1 km cdl data extracted with pump coordinates (this operation has no practical use)
def compare_pumpcrop_cdlcrop(pump_csv, cdl_dir_100m=r'.\Yearly_CDL_100m_July', output_csv='pump_crop_match.csv',
                             yearlist=[2014, 2015, 2016, 2017, 2018, 2019]):
    cdl_dict = {
        1: 'Corn',
        2: 'Cotton',
        3: 'Rice',
        5: 'Soybeans',
        92: 'Catfish',
        0: 'Other'
    }
    pumps = pd.read_csv(pump_csv)
    cdl_rasters = glob(os.path.join(cdl_dir_100m, '*tif'))

    lon = pumps['Longitude'].tolist()
    lat = pumps['Latitude'].tolist()
    cdl_arr, cdl_file = read_raster_as_arr(cdl_rasters[0])
    transformer = Transformer.from_crs('EPSG:4326', cdl_file.crs.to_string(), always_xy=True)
    lon, lat = transformer.transform(lon, lat)
    pumps['Lon_new'] = lon
    pumps['Lat_new'] = lat

    pump_new = pd.DataFrame()
    for year in yearlist:
        pumps_year = pumps[pumps['ReportYear'] == year].copy()
        cdl_raster = None
        for cdl in cdl_rasters:
            if str(year) in cdl:
                cdl_raster = cdl
        cdl_arr, cdl_file = read_raster_as_arr(cdl_raster)

        crop_value_list = []
        crop_name_list = []
        for lon, lat in zip(pumps_year['Lon_new'], pumps_year['Lat_new']):
            row, col = cdl_file.index(lon, lat)
            crop_value = cdl_arr[row, col]
            crop_value_list.append(crop_value)
            crop_name_list = [cdl_dict.get(crop, np.nan) for crop in crop_value_list]

        pumps_year['crop_from_cdl'] = crop_name_list
        pump_new = pump_new.append(pumps_year)
    pump_new = pump_new.drop(columns=['Lon_new', 'Lat_new'])
    pump_new.to_csv(output_csv, index=False)

    match_crop_value = len(pump_new[pump_new['Crop(s)'] == pump_new['crop_from_cdl']])
    non_null_value = len(pump_new[pd.notnull(pump_new['crop_from_cdl'])])
    print(match_crop_value, non_null_value)
    print('% match in crop type {match}'.format(match=round(match_crop_value * 100 / non_null_value, 2)))


def relate_well_field(pump_csv, field_shape, output_csv, lat_pump='Latitude', lon_pump='Longitude', lat_field='Cent_y',
                      lon_field='Cent_x', year_filter=False, year_col='ReportYear', yearlist=[2019],
                      other_col='PermitNumb', coords_to_projected=True):
    """
    Find nearest field for each pumping well.

    :param pump_csv: Pumping well csv.
    :param field_shape: Field polygon shapefile with centroid coordinates.
    :param output_csv: Output csv.
    :param lat_pump: Latitude column in pump_csv.
    :param lon_pump: Longitude column in pump_csv.
    :param lat_field: Latitude column in field shapefile.
    :param lon_field: Longitude column in field shapefile.
    :param year_filter: Set to True if code operation has to be done for only selected years.
    :param year_col: Year column in pump_csv.
    :param yearlist: List of years to do code operations on. Only works if year_filter is set to True.
    :param other_col:
    :param coords_to_projected: Set to False if coordinates are already in projected system and conversion from
           geographic to projected is not necessary.
    :return: A joined dataframe of pumps and nearest fields.
    """

    pumps = pd.read_csv(pump_csv)
    fields = gpd.read_file(field_shape)

    pumps_coords = pumps[[lat_pump, lon_pump]]
    fields_coords = fields[[lat_field, lon_field]]

    if coords_to_projected:
        latitude_pump = pumps_coords[lat_pump].tolist()
        longitude_pump = pumps_coords[lon_pump].tolist()

        transformer = Transformer.from_crs('EPSG:4326', fields.crs.to_string(), always_xy=True)
        lon_pump, lat_pump = transformer.transform(longitude_pump, latitude_pump)
        pumps_coords = pd.DataFrame()
        pumps_coords['lat_pump'] = pd.Series(lat_pump)
        pumps_coords['lon_pump'] = pd.Series(lon_pump)

    kdtree_classifier = KDTree(fields_coords.values, metric='euclidean')
    distances, indices = kdtree_classifier.query(pumps_coords, k=1)
    distances = pd.Series(distances.flatten())

    indices_list = pd.Series(indices.flatten()).tolist()  # numpy 2d array to 1d array to pandas series conversion
    nearest_fields = fields.iloc[indices_list]
    nearest_fields = nearest_fields.reset_index()
    nearest_fields = nearest_fields[[lat_field, lon_field, other_col]]

    pumps_new = pumps
    new_df = pumps_new.join(nearest_fields, how='outer')
    new_df['distance'] = distances
    if year_filter:
        new_df = new_df[new_df[year_col].isin(yearlist)]
    new_df = new_df.reset_index(drop=True)

    # Storing nearest field coordinates as WGS 1984 projection
    field_lat = new_df[lat_field].tolist()
    field_lon = new_df[lon_field].tolist()

    transformer = Transformer.from_crs(fields.crs.to_string(), 'EPSG:4326', always_xy=True)
    field_lon, field_lat = transformer.transform(field_lon, field_lat)

    new_df = new_df.drop(columns={lat_field, lon_field}, axis=1)
    new_df['Cent_y'] = pd.Series(field_lat)
    new_df['Cent_x'] = pd.Series(field_lon)

    new_df.to_csv(output_csv, index=False)

    return new_df


# # 2019 to 2019 unique field polygons (Works)
# relate_well_field(pump_csv=vmp, field_shape=r'.\well_field_connection\fieldsof2019_with_crop.shp',
#                   output_csv=r'.\well_field_connection\old_connected_files\joined_2019_kdtree.csv', year_filter=True,
#                   yearlist=[2019], coords_to_projected=True, lat_field='Cent_y', lon_field='Cent_x')


def relate_well_field_includecrop(pump_csv, field_shape, output_csv, lat_pump='Latitude', lon_pump='Longitude',
                                  lat_field='Cent_y', lon_field='Cent_x', yearlist=[2014, 2015, 2016, 2017, 2018, 2019],
                                  coords_to_projected=True, n_neighbors=5, field_permit_attr='PermitNumb'):
    """
    Find nearest field for each pumping well.

    :param pump_csv: Pumping well csv.
    :param field_shape: Field polygon shapefile with centroid coordinates.
    :param output_csv: Output csv.
    :param lat_pump: Latitude column in pump_csv.
    :param lon_pump: Longitude column in pump_csv.
    :param lat_field: Latitude column in field shapefile.
    :param lon_field: Longitude column in field shapefile.
    :param yearlist: List of years to do code operations on. Only works if year_filter is set to True.
    :param coords_to_projected: Set to False if coordinates are already in projected system and conversion from
           geographic to projected is not necessary.
    :param field_permit_attr: Permit number attribute in field shapefile.
    :param n_neighbors: Number of neighest neighbor to calculate in kdtree process. Default set to 5.

    :return: A joined dataframe of pumps and nearest fields.
    """
    # reading files
    pumps = pd.read_csv(pump_csv)
    fields = gpd.read_file(field_shape)

    # selecting coordinates columns
    pumps = pumps[(pumps['ReportYear'].isin(yearlist)) & (pumps['AF_Acre'] != 0)]
    pumps_coords = pumps[[lat_pump, lon_pump]]
    fields_coords = fields[[lat_field, lon_field]]

    # converting coordinates to projected system
    if coords_to_projected:
        latitude_pump = pumps_coords[lat_pump].tolist()
        longitude_pump = pumps_coords[lon_pump].tolist()
        transformer = Transformer.from_crs('EPSG:4326', fields.crs.to_string(), always_xy=True)
        lon_pump, lat_pump = transformer.transform(longitude_pump, latitude_pump, )
        pumps_coords = pd.DataFrame()
        pumps_coords['lat_pump'] = pd.Series(lat_pump)
        pumps_coords['lon_pump'] = pd.Series(lon_pump)

    # kdtree operation
    kdtree_classifier = KDTree(fields_coords.values, metric='euclidean')
    distances, indices = kdtree_classifier.query(pumps_coords, k=n_neighbors)
    distances = pd.Series(distances.flatten())
    indices_list = pd.Series(indices.flatten()).tolist()  # numpy 2d array to 1d array to pandas series conversion

    # selecting nearest fields based on kdtree indices
    nearest_fields = fields.iloc[indices_list].copy()
    crop_columns_label = []
    for year in yearlist:
        crop_columns_label.append('crop_' + str(year))

    pumps_new = pumps[['Permit Number', 'County', 'Latitude', 'Longitude', 'Units', 'Acre Feet', 'Crop(s)',
                       'Acreage Total', 'AF_Acre', 'ReportYear']].copy()
    # correcting crop names in pumps database
    pump_crops = list(pumps_new['Crop(s)'])
    pump_crops_modified = []
    for crop in pump_crops:
        if crop == 'Fish Culture':
            pump_crops_modified.append('Catfish')
        elif crop == 'Peanuts':
            pump_crops_modified.append('Other')
        elif crop == 'Soybean':
            pump_crops_modified.append('Soybeans')
        else:
            pump_crops_modified.append(crop)
    pumps_new['Crop(s)'] = pump_crops_modified

    # creating a new pump index column (used in coparison later)
    pumps_new['pump_index'] = pumps_new.index
    pumps_index = list(pumps_new.index)

    # creating a new field index column in nearest field dataframe which has equal value to each corresponding
    # nearest pump index
    nearest_fields_new_index = []
    for i in pumps_index:
        nearest_fields_new_index.append(list(itertools.repeat(i, n_neighbors)))

    nearest_fields_new_index = list(itertools.chain(*nearest_fields_new_index))
    nearest_fields['field_index'] = None
    nearest_fields.loc[:, 'field_index'] = nearest_fields_new_index
    nearest_fields = nearest_fields.reset_index()
    nearest_fields = nearest_fields[[lat_field, lon_field, 'field_index', field_permit_attr, 'area_acre'] +
                                    crop_columns_label]

    # merging pump data and nearest field data
    new_df = pumps_new.merge(nearest_fields, left_on='pump_index', right_on='field_index')
    new_df['distance_kdtree'] = distances
    new_df = new_df.reset_index(drop=True)
    new_df.to_csv('kdtree.csv', index=False)

    # Storing nearest field coordinates as WGS 1984 projection
    field_lat = new_df[lat_field].tolist()
    field_lon = new_df[lon_field].tolist()
    transformer = Transformer.from_crs(fields.crs.to_string(), 'EPSG:4326', always_xy=True)
    field_lon, field_lat = transformer.transform(field_lon, field_lat)
    new_df = new_df.drop(columns={lat_field, lon_field}, axis=1)
    new_df['Cent_y'] = pd.Series(field_lat)
    new_df['Cent_x'] = pd.Series(field_lon)

    # selecting nearest fields based on matching crop type for each year
    unique_index = new_df['pump_index'].unique()
    final_index_list = []
    crop_from_cdl = []
    for year in yearlist:
        year_df = new_df[new_df['ReportYear'] == year]
        year_df = year_df[['Permit Number', 'County', 'Latitude', 'Longitude', 'Units', 'Acre Feet', 'Crop(s)',
                           'Acreage Total', 'AF_Acre', 'ReportYear', 'pump_index', 'field_index', 'PermitNumb',
                           'crop_' + str(year), 'area_acre', 'distance_kdtree', 'Cent_y', 'Cent_x']]
        print('Looking for match in', str(year))
        for index_match in unique_index:
            year_df2 = year_df[year_df['pump_index'] == index_match]
            for index, row in year_df2.iterrows():
                if row['Crop(s)'].lower() == row['crop_' + str(year)].lower():
                    crop_from_cdl.append(row['crop_' + str(year)])
                    final_index_list.append(index)
                    break
    final_df = new_df.iloc[final_index_list]
    final_df = final_df[['Permit Number', 'County', 'Latitude', 'Longitude', 'Units', 'Acre Feet', 'Crop(s)',
                         'Acreage Total', 'AF_Acre', 'ReportYear', 'pump_index', 'field_index', 'PermitNumb',
                         'area_acre', 'distance_kdtree', 'Cent_y', 'Cent_x']]
    final_df['crop_from_field'] = crop_from_cdl
    final_df = final_df.reset_index(drop=True)
    final_df.to_csv(output_csv, index=False)
    return final_df


# # 2014_2019
# field_with_crop = r'H:\USGS_MAP\Fahim\well_field_connection\Permit_Boundaries_with_crop.shp'
# new_csv = r'.\well_field_connection\new_connected_files\joined_2014_2019_kdtree.csv'
# relate_well_field_includecrop(pump_csv=vmp, field_shape=field_with_crop, output_csv=new_csv,
#                               yearlist=[2014, 2015, 2016, 2017, 2018, 2019],
#                               coords_to_projected=True)

# # 2019 to 2019 unique field polygons (Works)
# relate_well_field_includecrop(pump_csv=vmp, field_shape=r'.\well_field_connection\fieldsof2019_with_crop.shp',
#                               output_csv=r'.\well_field_connection\new_connected_files\joined_2019_kdtree.csv',
#                               yearlist=[2019], coords_to_projected=True, lat_field='Cent_y', lon_field='Cent_x',
#                               n_neighbors=5)


def extracting_crop_data(input_shape, cdl_dir, output_shape, yearlist=[2014, 2015, 2016, 2017, 2018, 2019]):
    fields = gpd.read_file(input_shape)
    cdl_rasters = glob(os.path.join(cdl_dir, '*tif'))
    cdl_dict = {
        1: 'Corn',
        2: 'Cotton',
        3: 'Rice',
        5: 'Soybeans',
        92: 'Catfish',
        0: 'Other'
    }
    cdl_arr, cdl_file = read_raster_as_arr(cdl_rasters[0])
    if fields.crs != cdl_file.crs:
        fields = fields.to_crs(cdl_file.crs)

    def getXY(pt):
        return pt.x, pt.y

    centroidseries = fields['geometry'].centroid

    lon, lat = [list(t) for t in zip(*map(getXY, centroidseries))]
    fields['Cent_x'] = pd.Series(lon)
    fields['Cent_y'] = pd.Series(lat)

    crop_columns_labels = []
    for year in yearlist:
        print('Extracting crop value for', str(year), '...')
        crop_columns_labels.append('crop_' + str(year))
        for cdl in cdl_rasters:
            if str(year) in cdl:
                cdl_arr, cdl_file = read_raster_as_arr(cdl)
                crop_value = []
                for lon, lat in zip(fields['Cent_x'], fields['Cent_y']):
                    row, col = cdl_file.index(lon, lat)
                    if (row <= cdl_file.shape[0]) & (col <= cdl_file.shape[1]):
                        value = cdl_arr[row, col]
                        crop_value.append(value)
                crop_column = 'crop_' + str(year)
                crop_name_list = [cdl_dict.get(crop, np.nan) for crop in crop_value]
                fields[crop_column] = pd.Series(crop_name_list)

    fields = fields.dropna(axis=0, subset=crop_columns_labels)
    fields['area_acre'] = round(fields.area * 0.000247105, 0)  # area in acre
    fields.to_file(output_shape)

    return output_shape

# # 2014_2019 cropdata extraction
# cropfield=extracting_crop_data(input_shape=fields_all, cdl_dir=r'..\Outputs\CDL_AIWUM1',
#                      output_shape=r'.\well_field_connection\Permit_Boundaries_with_crop.shp',
#                      yearlist=[2014, 2015, 2016, 2017, 2018, 2019])


# # 2019 cropdata extraction
# cropfield = extracting_crop_data(input_shape=field_19, cdl_dir=r'.\Yearly_CDL_100m_July',
#                      output_shape=r'.\well_field_connection\fieldsof2019_with_crop.shp',
#                      yearlist=[2019])

# # 2019 Valiation Test
# field19 = field19.drop_duplicates('PermitNumb').sort_index()  # Duplicate permit number exists in 2019 unique polygons
# pumps19 = pump[pump['ReportYear'] == 2019]
#
# pumps_new = pd.merge(pumps19, field19, how='inner', left_on='Permit Number', right_on='PermitNumb')
# pumps_new= pumps_new[
#     ['Permit Number', 'County_x', 'AF_Acre', 'Crop(s)', 'ReportYear', 'Latitude_x', 'Longitude_x', 'PermitNumb',
#      'Cent_Y', 'Cent_X']]
# pumps_new = pumps_new.rename(columns={'County_x': 'County', 'Latitude_x': 'Lat_pump', 'Longitude_x': 'Lon_pump',
#                                     'Cent_Y': 'Lat_field', 'Cent_X': 'Lon_field'})
#
# #Converting to projected coordinate system
# latitude_pump = pumps_new['Lat_pump'].tolist()
# longitude_pump = pumps_new['Lon_pump'].tolist()
# latitude_field = pumps_new['Lat_field'].tolist()
# longitude_field = pumps_new['Lon_field'].tolist()
# new_lat_pump = []
# new_lon_pump = []
# new_lat_field = []
# new_lon_field = []
# for each in zip(latitude_field, longitude_field):
#     lat, lon = transform.transform('EPSG:4326', 'EPSG:3857', [each[0]], [each[1]])
#     new_lat_field.append(lat[0])
#     new_lon_field.append(lon[0])
#     pumps_new['Lat_field_projected'] = pd.Series(new_lat_field)
#     pumps_new['Lon_field_projected'] = pd.Series(new_lon_field)
#
# kdtree_2019=pd.read_csv(r'.\well_filed_connection\joined_2019_kdtree.csv')
# kdtree_2019=kdtree_2019[['Permit Number', 'Cent_Y','Cent_X']]
# pumps_fields=pumps_new.merge(kdtree_2019, how='inner', on='Permit Number')
# pumps_fields=pumps_fields.rename(columns={'Cent_Y':'Lat_field_kdtree','Cent_X':'Lon_field_kdtree'})
#
# #Changing kdtree's coordinate system to projected
# lat_field_kdtree = pumps_fields['Lat_field_kdtree'].tolist()
# lon_field_kdtree = pumps_fields['Lon_field_kdtree'].tolist()
# kdtree_lat = []
# kdtree_lon = []
# for each in zip(lat_field_kdtree, lon_field_kdtree):
#     lat, lon = transform.transform('EPSG:4326', 'EPSG:3857', [each[0]], [each[1]])
#     kdtree_lat.append(lat[0])
#     kdtree_lon.append(lon[0])
#     pumps_fields['Lat_field_kdtree_projected'] = pd.Series(kdtree_lat)
#     pumps_fields['Lon_field_kdtree_projected'] = pd.Series(kdtree_lon)
#
# for index, row in pumps_fields.iterrows():
#     pumps_fields.loc[index,'distance_bet_centroids'] = sqrt((row['Lon_field_projected'] - row['Lon_field_kdtree_projected']) ** 2 +
#                                                           (row['Lat_field_projected'] - row['Lat_field_kdtree_projected']) ** 2)
# pumps_fields=pumps_fields.drop( columns={'Lat_field_projected', 'Lon_field_projected', 'Lat_field_kdtree_projected',
#                                          'Lon_field_kdtree_projected'}, axis=1)
# pumps_fields.to_csv(r'.\well_filed_connection\joined_2019_permitNumber.csv')


# # 2019 Valiation Test (New)
# field_19_withcrop = gpd.read_file(r'.\well_field_connection\fieldsof2019_with_crop.shp')
# field19 = field_19_withcrop.drop_duplicates('PermitNumb').sort_index()  # Duplicate permit number exists in 2019 unique polygons
# vmp = pd.read_csv(vmp)
# pumps19 = vmp[vmp['ReportYear'] == 2019].copy()
#
# # Modifying correct crop type
# pump_crops = list(pumps19['Crop(s)'])
# pump_crops_modified = []
# for crop in pump_crops:
#     if crop == 'Fish Culture':
#         pump_crops_modified.append('Catfish')
#     elif crop == 'Peanuts':
#         pump_crops_modified.append('Other')
#     elif crop == 'Soybean':
#         pump_crops_modified.append('Soybeans')
#     else:
#         pump_crops_modified.append(crop)
# pumps19['Crop(s)'] = pump_crops_modified
#
# pumps_new = pd.merge(pumps19, field19, how='inner', left_on='Permit Number', right_on='PermitNumb')
# pumps_new = pumps_new[
#     ['Permit Number', 'County_x', 'AF_Acre', 'Crop(s)', 'ReportYear', 'Latitude_x', 'Longitude_x', 'PermitNumb',
#      'Cent_y', 'Cent_x', 'crop_2019']]
# pumps_new = pumps_new.rename(columns={'County_x': 'County', 'Latitude_x': 'Lat_pump', 'Longitude_x': 'Lon_pump',
#                                       'Cent_y': 'Lat_field', 'Cent_x': 'Lon_field'})
#
# # Converting Pump Latitude and Longtitude coordinates from WGS 1984 to Projected
# latitude_pump = pumps_new['Lat_pump'].tolist()
# longitude_pump = pumps_new['Lon_pump'].tolist()
# transformer = Transformer.from_crs('EPSG:4326', field19.crs.to_string(), always_xy=True)
# lon_pump, lat_pump = transformer.transform(longitude_pump, latitude_pump, )
# pumps_new['Lat_pump'] = pd.Series(lat_pump)
# pumps_new['Lon_pump'] = pd.Series(lon_pump)
#
#
# kdtree_2019 = pd.read_csv(r'.\well_field_connection\new_connected_files\joined_2019_kdtree.csv')
# kdtree_2019 = kdtree_2019[['Permit Number', 'Cent_y', 'Cent_x']]
# pumps_fields = pumps_new.merge(kdtree_2019, how='inner', on='Permit Number')
# pumps_fields = pumps_fields.rename(columns={'Cent_y': 'Lat_field_kdtree', 'Cent_x': 'Lon_field_kdtree'})
#
# # Changing kdtree's coordinate system to projected
# lat_field_kdtree = pumps_fields['Lat_field_kdtree'].tolist()
# lon_field_kdtree = pumps_fields['Lon_field_kdtree'].tolist()
# transformer = Transformer.from_crs('EPSG:4326', field19.crs.to_string(), always_xy=True)
# lon_field_kdtree, lat_field_kdtree = transformer.transform(lon_field_kdtree, lat_field_kdtree)
# pumps_fields['Lat_field_kdtree'] = pd.Series(lat_field_kdtree)
# pumps_fields['Lon_field_kdtree'] = pd.Series(lon_field_kdtree)
#
# for index, row in pumps_fields.iterrows():
#     pumps_fields.loc[index, 'distance_bet_centroids'] = sqrt(
#         (row['Lon_field'] - row['Lon_field_kdtree']) ** 2 +
#         (row['Lat_field'] - row['Lat_field_kdtree']) ** 2)
#
# pumps_fields.to_csv(r'.\well_field_connection\new_connected_files\joined_2019_PermitNumber.csv')