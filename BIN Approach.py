import os
from glob import glob
import numpy as np
import pandas as pd
from statistics import mean
from datetime import datetime
import gdal
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from Python_Files.maplibs.rasterops import read_raster_as_arr

pd.set_option('display.max_columns', 500)


def create_precip_bin(output_csv, pumping_precip_dir='../RealtimeMeterNetwork/Site_Rainfall_Data',
                      bins=[-np.inf, 0.05, 0.1, 0.3, 0.5, 0.75, 1, 1.1, np.inf],
                      labels=['0-0.05', '0.05-0.1', '0.1-0.3', '0.3-0.5', '0.5-0.75', '0.75-1', '1-1.1', '>1.1']):
    """
    creates a csv of average 2-day, 3-day,...7-day pumping for rainfall bins (i.e. 0-0.5 inch, 0.5-1 inch...)

    :param output_csv: FIlepath of output csv.
    :param pumping_precip_dir: Directory path of sites' csv file with both pumping and precipitation data.
    :param bins: List of rainfall bins. Default set to [-np.inf, 0.1, 0.3, 0.5, 0.75, 1, 1.1, np.inf].
    :param labels: List of labels of rainfall bins. Default set to ['0-0.1', '0.1-0.3', '0.3-0.5', '0.5-0.75', '0.75-1',
                   '1-1.1', '>1.1']
    :return: A csv file with average 2-day, 3-day,...7-day pumping for rainfall bins.
    """
    pumping_precip_csv = glob(os.path.join(pumping_precip_dir, '*', '*with_obs_precip.csv'))

    total_df = pd.DataFrame()
    for each in pumping_precip_csv:
        df = pd.read_csv(each)
        pumping_inch = df['Pumping(in)'].to_list()
        df['rainfall_bin'] = pd.cut(x=df['Observed_precip(in)'], bins=bins, labels=labels)
        avg_2_day_pumping = []
        avg_3_day_pumping = []
        avg_4_day_pumping = []
        avg_5_day_pumping = []
        avg_6_day_pumping = []
        avg_7_day_pumping = []
        for i in range(0, len(pumping_inch)):
            avg_2_day = mean(pumping_inch[i:i + 2])
            avg_2_day_pumping.append(avg_2_day)
            avg_3_day = mean(pumping_inch[i:i + 3])
            avg_3_day_pumping.append(avg_3_day)
            avg_4_day = mean(pumping_inch[i:i + 4])
            avg_4_day_pumping.append(avg_4_day)
            avg_5_day = mean(pumping_inch[i:i + 5])
            avg_5_day_pumping.append(avg_5_day)
            avg_6_day = mean(pumping_inch[i:i + 6])
            avg_6_day_pumping.append(avg_6_day)
            avg_7_day = mean(pumping_inch[i:i + 7])
            avg_7_day_pumping.append(avg_7_day)
        df['avg_2_day_pumping'] = avg_2_day_pumping
        df['avg_3_day_pumping'] = avg_3_day_pumping
        df['avg_4_day_pumping'] = avg_4_day_pumping
        df['avg_5_day_pumping'] = avg_5_day_pumping
        df['avg_6_day_pumping'] = avg_6_day_pumping
        df['avg_7_day_pumping'] = avg_7_day_pumping
        df = df[:-7]
        total_df = total_df.append(df)
    total_df.to_csv('../RealtimeMeternetwork/Bin_precip.csv', index=False)
    bin_average = total_df.groupby('rainfall_bin')[['avg_2_day_pumping', 'avg_3_day_pumping', 'avg_4_day_pumping',
                                                    'avg_5_day_pumping', 'avg_6_day_pumping',
                                                    'avg_7_day_pumping']].mean()
    bin_average.to_csv(output_csv)


# create_precip_bin(output_csv='../RealtimeMeterNetwork/Bin_precip_avg_06.csv',
#                   bins=[-np.inf, 0, 0.02, 0.051, 0.1, 0.3, 0.5, 0.75, 1, 1.1, np.inf],
#                   labels=['0', '0-0.02', '0.02-0.051', '0.051-0.1', '0.1-0.3', '0.3-0.5', '0.5-0.75', '0.75-1', '1-1.1',
#                           '>1.1'])


def save_prism_bil_data_as_tif(prism_rainfall_data_dir=r'../Daily_Precipitation_data_Prism_2014_2021',
                               final_tif_data_dir=r'../Daily_Precipitation_data_Prism_2014_2021'
                                                  r'/all_daily_rainfall_data_tif'):
    """
    Convert daily prism rainfall data from .bil format to GeoTiff format.

    :param prism_rainfall_data_dir: Daily prism rainfall data (as .bil) directory.
    :param final_tif_data_dir: Output Daily prism rainfall data (as GeoTiff) directory.
    :return: None.
    """
    rainfall_datasets = glob(os.path.join(prism_rainfall_data_dir, '*', '*.bil'))
    for data in rainfall_datasets:
        data_name = data[data.rfind(os.sep) + 1:data.rfind('_')] + '.tif'
        dest_data = os.path.join(final_tif_data_dir, data_name)
        gdal.Translate(dest_data, data, format='GTiff', outputSRS='EPSG:4326')
        # converted to WGS84 from the original NAD83 because there is slight difference between the measurement between
        # the two projections. Keeping the data in NAD83 will require projection system conversion for well points
        # (from wgs84 to NAD83) which may change the location of wells slightly


def create_daily_rainfall_dictionary(prism_rainfall_directory=r'../Daily_Precipitation_data_Prism_2014_2021'
                                                              r'/all_daily_rainfall_data_tif'):
    """
    create a dictionary with date strings (April-September) as key and their corresponding prism daily rainfall data
    (numpy array and rasterio file) as dictionary item.

    :param prism_rainfall_directory: Daily prism rainfall data directory.
    :return: A dictionary with daily prism rainfall data (numpy array and rasterio file).
    """
    prism_daily_rainfall_datasets = sorted(glob(os.path.join(prism_rainfall_directory, '*.tif')))
    daily_rainfall_dict = {}

    for data in prism_daily_rainfall_datasets:
        date_str = data[data.rfind('_') + 1:data.rfind('.')]
        date = datetime.strptime(date_str, '%Y%m%d')
        month = date.month
        if 4 <= month <= 9:
            date = datetime.strftime(date, '%Y%m%d')
            daily_rainfall_dict[date] = read_raster_as_arr(data)
    return daily_rainfall_dict


def create_annual_prism_effective_rainfall_csv(annual_dataset=r'../../Data/main/Annual_Data_Latest.csv',
                                               low_rainfall_in=0.1, high_rainfall_in=1.1,
                                               output_csv='../RealtimeMeterNetwork/csv_effective_rainfall_filter'
                                                          '/low01_high11.csv'):
    """
    Create a csv with *annual effective rainfall (in), **annual rainfall prism (in) and ***annual pumping (in) for
    growing season (April to September).
    *annual effective rainfall (in) was summed from daily effective rainfall (after filtering out excess rainfall from
                                                                              daily prism rainfall data)
    **annual rainfall prism (in) was summed from daily prism rainfall data.
    ***annual pumping (in) from USGS annual dataset.

    :param annual_dataset: USGS Annual Dataset with annual pumping data.
    :param low_rainfall_in: lower range of rainfall less than which effective rainfall will be zero inch.
    :param high_rainfall_in: highest range of rainfall more than which rainfall will be turned into runoff.
    :param output_csv: output csv filepath.
    :return: output csv filepath.
    """
    prism_daily_rainfall_dict = create_daily_rainfall_dictionary()

    annual_df = pd.read_csv(annual_dataset)
    annual_df['year_str'] = annual_df['Year'].astype(str)

    annual_rainfall_inch_list = []
    effective_annual_rainfall_list = []

    for index, row in annual_df.iterrows():
        daily_rainfall_list = []
        effective_daily_rainfall_list = []
        year = row['year_str']
        lon, lat = row['Longitude'], row['Latitude']
        for key in prism_daily_rainfall_dict.keys():
            if year in key:
                rain_arr, rain_file = prism_daily_rainfall_dict[key][0], prism_daily_rainfall_dict[key][1]
                row, col = rain_file.index(lon, lat)
                daily_rainfall_mm = rain_arr[row, col]
                daily_rainfall_inch = daily_rainfall_mm / 25.4
                daily_rainfall_list.append(daily_rainfall_inch)

                if daily_rainfall_inch < low_rainfall_in:
                    eff_rainfall_inch = 0
                elif daily_rainfall_inch > high_rainfall_in:
                    eff_rainfall_inch = high_rainfall_in
                else:
                    eff_rainfall_inch = daily_rainfall_inch
                effective_daily_rainfall_list.append(eff_rainfall_inch)

        total_annual_rainfall_inch = sum(daily_rainfall_list)
        annual_rainfall_inch_list.append(total_annual_rainfall_inch)

        total_effective_rainfall_inch = sum(effective_daily_rainfall_list)
        effective_annual_rainfall_list.append(total_effective_rainfall_inch)

    df_ml = annual_df[['Latitude', 'Longitude', 'AF/Acre', 'ppt']]
    df_ml['annual_pumping(in)'] = df_ml['AF/Acre'] * 12
    df_ml['rainfall_original(in)'] = df_ml['ppt'] / 25.4

    df_ml['rainfall_prism(in)'] = annual_rainfall_inch_list
    df_ml['effecttive_rainfall(in)'] = effective_annual_rainfall_list
    print(df_ml)
    df_ml.to_csv(output_csv, index=False)

    return output_csv


# create_annual_prism_effective_rainfall_csv(low_rainfall_in=0.051, high_rainfall_in=1.1,
#                                            output_csv='../RealtimeMeterNetwork/csv_effective_rainfall_filter'
#                                                       '/low0051_high11.csv')

# Regression model for Annual Original Rainfall and Annual Effective Rainfall with Annual Pumping
# df_ml = pd.read_csv('../RealtimeMeterNetwork/csv_effective_rainfall_filter/low0051_high11.csv')
# X = df_ml['rainfall_prism(in)'].values.reshape(-1, 1)
# y = df_ml['annual_pumping(in)'].values
# lr_model = LinearRegression()
# lr_model.fit(X, y)
# y_pred = lr_model.predict(X)
# print('R2 score:', r2_score(y, y_pred))
# plt.scatter(X, y, color='g', alpha=0.3)
# plt.plot(X, y_pred, color='k')
# plt.xlabel('Annual Rainfall Prism (in)')
# plt.ylabel('Annual Pumping(in)')
# plt.show()
#
# X_effec = df_ml['effecttive_rainfall(in)'].values.reshape(-1, 1)
# y = df_ml['annual_pumping(in)'].values
# lr_model2 = LinearRegression()
# lr_model2.fit(X_effec, y)
# y_pred_effec = lr_model2.predict(X_effec)
# print('R2 score:', r2_score(y, y_pred_effec))
# plt.scatter(X_effec, y, color='g', alpha=0.3)
# plt.plot(X_effec, y_pred_effec, color='k')
# plt.xlabel('Annual Effective Rainfall (in)')
# plt.ylabel('Annual Pumping(in)')
# plt.show()
