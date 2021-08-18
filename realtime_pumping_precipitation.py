import os
from glob import glob
import rasterio
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statistics import mean
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# #Download data from URL
def download_precip_from_url(out_dir, start_day, end_day):
    """
    Download Daily Precipitation data (NCEP Stage IV Daily Accumulations) from
    'https://water.weather.gov/precip/download.php'.

    :param out_dir: Output directory where data will be downloaded.
    :param start_day: Start day for downloading data. Must be in '%Y/%m/%d' format. For example: '2019/06/01'
    :param end_day: End day for downloading data. Must be in '%Y/%m/%d' format. For example: '2019/06/01'

    :return: Data downloaded for the date range.
    """
    start_day_list = start_day.split('/')
    end_day_list = end_day.split('/')
    download_dir = out_dir + '_' + start_day_list[0] + start_day_list[1] + start_day_list[2] + '_' + end_day_list[0] + \
                   end_day_list[1] + end_day_list[2]
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    start_day = datetime.strptime(start_day, '%Y/%m/%d')
    end_day = datetime.strptime(end_day, '%Y/%m/%d')
    dates = [start_day + timedelta(days=x) for x in range((end_day - start_day).days + 1)]

    for date in dates:
        year = date.strftime('%Y')
        month = date.strftime('%m')
        day = date.strftime('%d')
        url_link = 'https://water.weather.gov/precip/downloads/' + year + '/' + month + '/' + day + '/' + \
                   'nws_precip_1day_' + year + month + day + '_conus.tif'
        print('Downloading Data for', year + '/' + month + '/' + day)
        filename = url_link[url_link.rfind('/') + 1:]
        down_fname = os.path.join(download_dir, filename)
        r = requests.get(url_link, allow_redirects=True)
        open(down_fname, 'wb').write(r.content)


# download_precip_from_url(out_dir=r'H:\USGS_MAP\Fahim\Daily_Precipitation_Data', start_day='2021/01/01',
#                          end_day='2021/06/30')

def download_site_data(site_info_excel='../RealtimeMeterNetwork/USGS_Data/StationNotes/MAP20XX_WaterUse_'
                                       'QuickReferenceSheet_12_08_2020.xlsx',
                       start_date='2017-01-01', end_date='2021-07-31',
                       download_dir='../RealtimeMeterNetwork/Site_Daily_Data',
                       description_download_dir='../RealtimeMeterNetwork/Site_Daily_Data/site_description_data'):
    """
    Download realtime site daily pumping data from USGS site.

    :param site_info_excel: Site info Excel file.
    :param start_date: Start Date for the daily data. **
    :param end_date: End date for the daily data. **
    :param download_dir: Download directory.
    :param description_download_dir: Directory for site description .txt download.
    ** (start/end date may/may not be available in the downloaded data based on availability)

    :return: Downloaded realtime site daily data.
    """

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    if not os.path.exists(description_download_dir):
        os.makedirs(description_download_dir)

    site_info_df = pd.read_excel(site_info_excel, sheet_name='2020')
    site_info_df = site_info_df[site_info_df['All data Present'].isin(['yes', 'yes EST need new transmitter',
                                                                       'yes/2 years', 'yes/ crawfish',
                                                                       'full year'])]

    site_number_list = site_info_df['Site_number'].tolist()
    for site in site_number_list:
        site_daily_data_link = 'https://waterdata.usgs.gov/nwis/dv?cb_72272=on&format=rdb&site_no=' + str(site) + \
                               '&referred_module=sw&period=&begin_date=' + start_date + '&end_date=' + end_date
        site_description_link = 'https://waterdata.usgs.gov/nwis/inventory?search_site_no=' + str(site) + \
                                '&search_site_no_match_type=exact&group_key=NONE&format=sitefile_output&sitefile_' \
                                'output_format=rdb&column_name=dec_lat_va&column_name=dec_long_va&list_of_search_' \
                                'criteria=search_site_no'
        print('Downloading Data for', site)
        daily_data_filename = str(site) + '.txt'
        down_fname = os.path.join(download_dir, daily_data_filename)
        r = requests.get(site_daily_data_link, allow_redirects=True)
        open(down_fname, 'wb').write(r.content)

        site_description_file_name = str(site) + '_site_description' + '.txt'
        down_fname = os.path.join(description_download_dir, site_description_file_name)
        r = requests.get(site_description_link, allow_redirects=True)
        open(down_fname, 'wb').write(r.content)


def get_acreage_data(pump_csv, permit_number, permit_number_col='Permit Number', year_col='ReportYear',
                     acre_col='Acreage Total'):
    """
    Get yearly acreage data from pump database.

    :param pump_csv: Map project yearly pump data database filepath.
    :param permit_number: Permit Number in 'MS-GW-47743' format.
    :param permit_number_col: Permit number column in pump database.
    :param year_col: Year column in pump database.
    :param acre_col: Acreage column in pump database.

    :return: Yearly pumped acreage value as acre_2014, acre_2015, acre_2016, acre_2017, acre_2018, acre_2019, acre_2020.
    """
    acre_2014 = None
    acre_2015 = None
    acre_2016 = None
    acre_2017 = None
    acre_2018 = None
    acre_2019 = None
    acre_2020 = None

    pump_df = pd.read_csv(pump_csv)
    permit_df = pump_df[pump_df[permit_number_col] == permit_number]
    for index, row in permit_df.iterrows():
        if row[year_col] == 2014:
            acre_2014 = row[acre_col]
        elif row[year_col] == 2015:
            acre_2015 = row[acre_col]
        elif row[year_col] == 2016:
            acre_2016 = row[acre_col]
        elif row[year_col] == 2017:
            acre_2017 = row[acre_col]
        elif row[year_col] == 2018:
            acre_2018 = row[acre_col]
        elif row[year_col] == 2019:
            acre_2019 = row[acre_col]
        elif row[year_col] == 2020:
            acre_2020 = row[acre_col]
    return acre_2014, acre_2015, acre_2016, acre_2017, acre_2018, acre_2019, acre_2020


# def read_save_site_daily_data(txtfilepath, outcsv_dir, lat_deg, lat_min, lat_sec, lon_deg, lon_min, lon_sec,
#                               permit_number, pump_csv=r'.\well_field_connection\VMP_readings_2014_2020.csv',
#                               skiprows=30):
#     """
#     Reading and Saving .txt daily data downloaded from USGS Real Time Server as csv file
#     (https://waterdata.usgs.gov/nwis/inventory?search_criteria=search_site_no&submitted_form=introduction).
#
#     :param pump_csv:
#     :param permit_number:
#     :param txtfilepath: File path of the daily data. Must be in .txt format.
#     :param outcsv_dir: Output directory path to save the daily data as csv file.
#     :param lat_deg: Latitude degree value (**).
#     :param lat_min: Latitude minute value (**).
#     :param lat_sec: Latitude second value (**).
#     :param lon_deg: Longitude degree value (**).
#     :param lon_min: Longitude minute value (**).
#     :param lon_sec: Longitude second value (**).
#     :param skiprows: Number of rows to skip while reading the file with pandas. Default set to 28 line.
#
#     ** Collect the Latitude Longtitude data from the USGS site.
#
#     :return: Daily pumping data saved as csv file.
#     """
#     df = pd.read_csv(txtfilepath, sep='\t', index_col=False, skiprows=skiprows)
#     df = df.iloc[:, 1:5]
#     df.columns = ['Site_ID', 'Date', 'Pumping(Acre-ft)', 'Approval']
#     df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
#     df['Latitude_NAD83'] = lat_deg + (lat_min / 60) + (lat_sec / 3600)
#     df['Longitude_NAD83'] = lon_deg + (lon_min / 60) + (lon_sec / 3600)
#     df = df[['Site_ID', 'Latitude_NAD83', 'Longitude_NAD83', 'Date', 'Approval', 'Pumping(Acre-ft)']]
#
#     # extracting yearly acreage data by the pump
#     acre_2014, acre_2015, acre_2016, acre_2017, acre_2018, acre_2019, acre_2020 = get_acreage_data(pump_csv,
#                                                                                                    permit_number)
#     # creating a column 'Pumping(in) in the database
#     df_ml = pd.DataFrame()
#     unique_years = df['Date'].dt.year.unique()
#     for year in unique_years:
#         year_df = df[df['Date'].dt.year == year].copy()
#
#         if year == 2014:
#             year_df['Pumping(in)'] = year_df['Pumping(Acre-ft)']*12/acre_2014
#         elif year == 2015:
#             year_df['Pumping(in)'] = year_df['Pumping(Acre-ft)']*12/acre_2015
#         elif year == 2016:
#             year_df['Pumping(in)'] = year_df['Pumping(Acre-ft)']*12/acre_2016
#         elif year == 2017:
#             year_df['Pumping(in)'] = year_df['Pumping(Acre-ft)'] * 12 / acre_2017
#         elif year == 2018:
#             year_df['Pumping(in)'] = year_df['Pumping(Acre-ft)'] * 12 / acre_2018
#         elif year == 2019:
#             year_df['Pumping(in)'] = year_df['Pumping(Acre-ft)'] * 12 / acre_2019
#         elif year == 2020:
#             year_df['Pumping(in)'] = year_df['Pumping(Acre-ft)'] * 12 / acre_2020
#
#         df_ml = df_ml.append(year_df)
#
#     df_ml['Pumping(mm)'] = df_ml['Pumping(in)']*25.4
#     df_ml['Permit Number'] = permit_number
#
#     csv_name = txtfilepath[txtfilepath.rfind('/') + 1:txtfilepath.rfind('.')] + '.csv'
#     df_ml.to_csv(os.path.join(outcsv_dir, csv_name), index=False)


# read_save_site_daily_data(txtfilepath='H:/USGS_MAP/Fahim/RealtimeMeterNetwork/Site_Daily_Data/332245090320901.txt',
#                           outcsv_dir='H:/USGS_MAP/Fahim/RealtimeMeterNetwork/Site_Daily_Data',
#                           lat_deg=33, lat_min=22, lat_sec=45, lon_deg=90, lon_min=32, lon_sec=9,
#                           permit_number='MS-GW-44991', skiprows=27)


def read_save_site_daily_data(outcsv_dir,
                              site_info_excel='../RealtimeMeterNetwork/USGS_Data/StationNotes/MAP20XX_WaterUse_'
                                              'QuickReferenceSheet_12_08_2020.xlsx', skiprows=30,
                              site_daily_data_dir='../RealtimeMeterNetwork/Site_Daily_Data',
                              site_description_dir='../RealtimeMeterNetwork/Site_Daily_Data/site_description_data'):
    """
    Reading and Saving .txt daily data downloaded from USGS Real Time Server as csv file
    (https://waterdata.usgs.gov/nwis/inventory?search_criteria=search_site_no&submitted_form=introduction).

    :param outcsv_dir: Output directory path to save the daily data as csv file.
    :param site_info_excel: Realtime site information excel.
    :param skiprows: Number of rows to skip while reading the file with pandas. Default set to 28 line.
    :param site_daily_data_dir: Directory path of pump station daily data.
    :param site_description_dir: Directory path of pump station description data.

    :return: Daily pumping data saved as csv file.
    """
    if not os.path.exists(outcsv_dir):
        os.makedirs(outcsv_dir)

    site_daily_data = glob(os.path.join(site_daily_data_dir, '*.txt'))
    for site in site_daily_data:
        df = pd.read_csv(site, sep='\t', index_col=False, skiprows=skiprows)
        df = df.iloc[:, 1:5]
        df.columns = ['Site_ID', 'Date', 'Pumping(Acre-ft)', 'Approval']
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

        site_number = site[site.rfind(os.sep) + 1:site.rfind('.')]
        site_description_txt = os.path.join(site_description_dir, str(site_number) + '_site_description.txt')
        description_df = pd.read_csv(site_description_txt, sep='\t', index_col=False, skiprows=24)
        description_df = description_df.iloc[:, 0:2]
        description_df.columns = ['Lat_WGS84', 'Lon_WGS84']
        Latitude = description_df['Lat_WGS84'][0]
        Longitude = description_df['Lon_WGS84'][0]

        df['Latitude_WGS84'] = Latitude
        df['Longitude_WGS84'] = Longitude
        site_id = df['Site_ID'][0]

        # adding acreage and crop information
        site_info_df = pd.read_excel(site_info_excel, sheet_name='2020')
        combined_acre = site_info_df[site_info_df['Site_number'] == site_id]['combined acres'].values[0]
        crop = site_info_df[site_info_df['Site_number'] == site_id]['crop'].values[0].lower()
        if crop in ['cotton', 'corn', 'soybeans', 'rice', 'catfish']:
            crop_mod = crop
        elif 'rice' in crop:
            crop_mod = 'rice'
        elif 'corn' in crop:
            crop_mod = 'corn'
        elif 'soybeans' in crop:
            crop_mod = 'soybeans'
        elif 'cotton' in crop:
            crop_mod = 'cotton'
        elif 'catfish' in crop:
            crop_mod = 'catfish'
        elif 'fingerling' or 'brood' or 'food fish' in crop:
            crop_mod = 'catfish'
        else:
            crop_mod = 'other'

        # creating a column 'Pumping(in) in the database
        df['Pumping(in)'] = df['Pumping(Acre-ft)'] * 12 / combined_acre
        df['Pumping(mm)'] = df['Pumping(in)'] * 25.4
        df['Acre'] = combined_acre
        df['crop'] = crop_mod

        csv_name = site[site.rfind(os.sep) + 1:site.rfind('.')] + '.csv'
        df.to_csv(os.path.join(outcsv_dir, csv_name), index=False)


def moving_average(input_series, days=7):
    """
    Calculates 7 day moving average for input series.
    :param input_series: Input pandas series.
    :param days: Number of days to calculate moving average on. Default set to 7.
    :return: A moving average series for input number of days.
    """
    avgd = np.convolve(input_series, np.ones(days), 'full') / days
    # # Ryan used in his code while mode was 'same'. May not be useful when mode is 'full'
    # clip_nans = int(1 + days / 2)
    # avgd[-clip_nans::] = np.nan
    # avgd[0:clip_nans] = np.nan
    return avgd


def extracting_daily_precipitation_in_pumping_record(pump_csv, precipitation_data_dir,
                                                     output_dir='./RealtimeMeterNetwork/Site_Rainfall_Data',
                                                     pump_date_col='Date'):
    """
    Extracts daily precipitation data (from NOAA precipitation data) in pump daily data (pumping record) csv.

    :param pump_csv: Daily pumping data csv filepath. Must contain a date column.
    :param precipitation_data_dir: Directory path of daily precipitation data.
    :param output_dir: Output dir filepath.
    :param pump_date_col: Date column in pump_csv.

    :return: A csv file containing daily precipitation data along with daily pumping data.
    """
    sep = pump_csv[pump_csv.rfind(os.sep)]
    if sep == os.sep:
        pump_csv = pump_csv.replace(sep, '/')

    pump_df = pd.read_csv(pump_csv)
    date_list = pump_df[pump_date_col].tolist()
    new_datestr_list = []
    for each in date_list:
        date_str = each.replace('-', '')
        new_datestr_list.append(date_str)
    pump_df['Date_str'] = pd.Series(new_datestr_list)

    precip_datasets = glob(os.path.join(precipitation_data_dir, '*.tif'))
    precip_data_dict = {}
    for each in precip_datasets:
        data_name = each[each.rfind(os.sep) + 1:]
        date_str = data_name.split('_')[3]
        precip_data_dict[date_str] = each
    precip_data = rasterio.open(precip_datasets[0])

    lon = pump_df['Longitude_WGS84'].tolist()
    lat = pump_df['Latitude_WGS84'].tolist()
    transformer = Transformer.from_crs('EPSG:4326', precip_data.crs.to_string(), always_xy=True)
    new_lon, new_lat = transformer.transform(lon, lat)

    # multiplied by -1 as the transformation couldn't convert to correct projection
    pump_df['Longitude_new'] = pd.Series(new_lon)
    pump_df['Latitude_new'] = pd.Series(new_lat)
    station_lon = pump_df['Longitude_new'][0]
    station_lat = pump_df['Latitude_new'][0]

    precip_value_list = []
    for index, row in pump_df.iterrows():
        precip_data = rasterio.open(precip_data_dict[row['Date_str']])
        precip_arr = precip_data.read(1)
        x, y = precip_data.index(station_lon, station_lat)
        precip_value = precip_arr[x, y]
        precip_value_list.append(precip_value)
    pump_df['Observed_precip(in)'] = pd.Series(precip_value_list)
    pump_df['Observed_precip(mm)'] = pd.Series(precip_value_list).multiply(25.4)
    pump_df = pump_df.drop('Date_str', axis=1)

    site_number = pump_csv[pump_csv.rfind('/') + 1:pump_csv.rfind('.')]
    final_output_dir = output_dir + '/' + site_number

    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)

    avg_pump = moving_average(pump_df['Pumping(mm)'], 7)
    avg_precip = moving_average(pump_df['Observed_precip(mm)'], 7)
    pump_df['7_day_avg_pumping(mm)'] = pd.Series(avg_pump)
    pump_df['7_day_avg_precip(mm)'] = pd.Series(avg_precip)

    output_csv = final_output_dir + '/' + site_number + '_with_obs_precip.csv'
    pump_df.to_csv(output_csv, index=False)

    return output_csv


def find_effective_rainfall(rainfall_value, percent=90):
    """
    This code calculates effective rainfall following US. Bureau of Reclamation's method.
    (might not be productive in our case).

    :param rainfall_value: Observed rainfall value in milimeter.
    :param percent: Confidence interval to use in interpolation. Default set to 90%.

    :return:Effective rainfall value in milimeter.
    """
    if 0 < rainfall_value < 25.4:
        percent_range = [90, 100]
        eff_rainfall_range = [22.9, 25.4]
        effective_rainfall = np.interp(percent, percent_range, eff_rainfall_range)
    elif 25.4 < rainfall_value < 50.8:
        percent_range = [85, 95]
        eff_rainfall_range = [44.4, 49.5]
        effective_rainfall = np.interp(percent, percent_range, eff_rainfall_range)
    elif 50.8 < rainfall_value < 76.2:
        percent_range = [75, 90]
        eff_rainfall_range = [63.5, 72.4]
        effective_rainfall = np.interp(percent, percent_range, eff_rainfall_range)
    elif 76.2 < rainfall_value < 101.6:
        percent_range = [50, 80]
        eff_rainfall_range = [76.2, 92.7]
        effective_rainfall = np.interp(percent, percent_range, eff_rainfall_range)
    elif 101.6 < rainfall_value < 127:
        percent_range = [30, 60]
        eff_rainfall_range = [83.8, 107.9]
        effective_rainfall = np.interp(percent, percent_range, eff_rainfall_range)
    elif 127 < rainfall_value < 152.4:
        percent_range = [10, 40]
        eff_rainfall_range = [86.4, 118.1]
        effective_rainfall = np.interp(percent, percent_range, eff_rainfall_range)
    else:
        percent_range = [0, 10]
        eff_rainfall_range = [86.4, 120.6]
        effective_rainfall = np.interp(percent, percent_range, eff_rainfall_range)

    return effective_rainfall


# Creating daily rainfall csv for pump coordinates
def create_tabular_daily_rainfall_for_pump(pumping_data_with_precip_csv,
                                           precip_data_dir='../Daily_Precipitation_Data_20160101_20210630'):
    """
    create a tabular daily precipitation record for pump location from daily precipitation data
    (precipitation data is in GeoTiff format).

    :param pumping_data_with_precip_csv: Pumping da
    :param precip_data_dir:

    :return: A csv file with daily precipitation record.
    """

    pump_df = pd.read_csv(pumping_data_with_precip_csv)
    daily_precip_datasets = glob(os.path.join(precip_data_dir, '*.tif'))
    precip_data_crs = rasterio.open(daily_precip_datasets[0]).crs.to_string()

    # converting station lon, lat to precipitation data's projection system
    lon = pump_df['Longitude_WGS84'][0]
    lat = pump_df['Latitude_WGS84'][0]
    transformer = Transformer.from_crs('EPSG:4326', precip_data_crs, always_xy=True)
    new_lon, new_lat = transformer.transform(lon, lat)

    daily_precip_df = pd.DataFrame()
    Observed_precip_in = []
    Date = []
    for each in daily_precip_datasets:
        precip_data = rasterio.open(each)
        precip_arr = precip_data.read(1)
        row, col = precip_data.index(new_lon, new_lat)
        precip_value = precip_arr[row, col]
        Observed_precip_in.append(precip_value)

        data_name = each[each.rfind(os.sep) + 1:]
        date_str = data_name.split('_')[3]
        date = datetime.strptime(date_str, '%Y%m%d')
        Date.append(date)

    daily_precip_df['Date'] = pd.Series(Date)
    daily_precip_df['Observed_precip(in)'] = pd.Series(Observed_precip_in)
    daily_precip_df['Date'] = pd.to_datetime(daily_precip_df['Date'])
    daily_precip_df['Observed_precip(mm)'] = daily_precip_df['Observed_precip(in)'] * 25.4

    csv_name = pumping_data_with_precip_csv[
               pumping_data_with_precip_csv.rfind('/') + 1:pumping_data_with_precip_csv.rfind('.')]
    station_id = csv_name.split('_')[0]
    outdir = pumping_data_with_precip_csv[:pumping_data_with_precip_csv.rfind('/')]
    output_csv = os.path.join(outdir, station_id + '_daily_precip.csv')
    daily_precip_df.to_csv(output_csv, index=False)

    return output_csv


# create_tabular_daily_rainfall_for_pump(pumping_data_with_precip_csv='./RealtimeMeterNetwork/Site_Daily_Data/332245090320901/332245090320901.csv')


def calculate_effective_rainfall(pumping_data_with_precip_csv,
                                 precip_data_dir='../Daily_Precipitation_Data_20160101_20210630'):
    """
    Calculated effective rainfall using US Bureau of Reclamation's method.

    :param pumping_data_with_precip_csv: Pump data csv with daily pumping and precipitation info.
    :param precip_data_dir: Daily precipitation data directory. Contains precipitation data in GeoTiff format.

    :return: A csv of pump data that contains effective rainfall value along with pumping record and observed rainfall
             data.
    """

    daily_rainfall_csv = create_tabular_daily_rainfall_for_pump(pumping_data_with_precip_csv, precip_data_dir)
    daily_precip_df = pd.read_csv(daily_rainfall_csv)

    daily_precip_df['Date'] = pd.to_datetime(daily_precip_df['Date'])
    daily_precip_df['Year'] = pd.DatetimeIndex(daily_precip_df['Date']).year
    yearlist = daily_precip_df['Year'].unique().tolist()
    monthlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    mean_monthly_precipitation = {}  # dictionary for mean monthly precipitation for each month for all the years
    monthly_precip_dict = {}  # created for storing monthly total precipitation for each year
    for months in monthlist:
        total_precip_list = []
        for years in yearlist:
            monthly_precip_df = daily_precip_df[(daily_precip_df['Date'].dt.month == months) &
                                                (daily_precip_df['Date'].dt.year == years)]
            total_monthly_precip = monthly_precip_df['Observed_precip(mm)'].sum()
            total_precip_list.append(total_monthly_precip)
            monthly_precip_dict[str(months) + '_' + str(years)] = total_monthly_precip
        mean_precip = sum(total_precip_list) / len(total_precip_list)
        mean_monthly_precipitation[months] = mean_precip

    monthly_effective_rainfall_dict = {}  # dictionary for storing effective monthly rainfall for each month
    for key, value in mean_monthly_precipitation.items():
        effective_rainfall = find_effective_rainfall(mean_monthly_precipitation[key])
        monthly_effective_rainfall_dict[key] = effective_rainfall

    pump_rainfall_df = pd.read_csv(pumping_data_with_precip_csv)
    pump_rainfall_df['Date'] = pd.to_datetime(pump_rainfall_df['Date'])
    pump_rainfall_df['Effective_rainfall(mm)'] = None
    Effective_rainfall_series = []
    for index, row in pump_rainfall_df.iterrows():
        month = row['Date'].month
        year = row['Date'].year
        month_year_key = str(month) + '_' + str(year)
        rainfall_perc = row['Observed_precip(mm)'] / monthly_precip_dict[month_year_key]
        eff_rainfall = rainfall_perc * monthly_effective_rainfall_dict[month]
        if row['Observed_precip(mm)'] < eff_rainfall:
            Effective_rainfall_series.append(row['Observed_precip(mm)'])
        else:
            Effective_rainfall_series.append(eff_rainfall)
    pump_rainfall_df['Effective_rainfall(mm)'] = pd.Series(Effective_rainfall_series)
    pump_rainfall_df['Effective_rainfall(in)'] = pump_rainfall_df['Effective_rainfall(mm)'] / 25.4

    csv_name = pumping_data_with_precip_csv[
               pumping_data_with_precip_csv.rfind('/') + 1:pumping_data_with_precip_csv.rfind('.')]
    station_id = csv_name.split('_')[0]
    outdir = pumping_data_with_precip_csv[:pumping_data_with_precip_csv.rfind('/')]
    output_csv = os.path.join(outdir, station_id + '_effective_precip.csv')
    pump_rainfall_df.to_csv(output_csv, index=False)


def plot_pumping_precipitation(pump_precip_csv, start_date1, end_date1, start_date2, end_date2, date_col='Date',
                               pumping_col='Pumping(mm)', precip_col='Observed_precip(mm)',
                               effec_precip_col='Effective_rainfall(mm)',
                               ax1_ylabel='Pumping (mm)', ax2_ylabel='Rainfall (mm)'):
    """
    Plotting Pumping and Precipitation (both Observed and Effective) value in a single plot for 2 years (Subplot set to
    2 for 2 years)

    :param pump_precip_csv: Csv with daily pumping and precipitation value.
    :param start_date1: Start date 1 for the plot. Format must be '2018-12-31'.
    :param end_date1: End date 1 for the plot. Format must be '2018-12-31'.
    :param start_date2: Start date 2 for the plot. Format must be '2018-12-31'.
    :param end_date2: End date for the plot. Format must be '2018-12-31'.
    :param date_col: Date column in the csv.
    :param pumping_col: Pumping value column in the csv.
    :param precip_col: Precipitation value column in the csv.
    :param effec_precip_col: Effective Precipitation value column in the csv.
    :param ax1_ylabel: ylabel for the left Y-axis. Default set to 'Pumping (mm)'.
    :param ax2_ylabel: ylabel for the lest axis. Default set to 'precipitation (mm)'.

    :return: A plot with Date/Year in the X-axis and Pumping & precipitation in the Y-axis.
    """

    pump_precip_df = pd.read_csv(pump_precip_csv)
    pump_precip_df[date_col] = pd.to_datetime(pump_precip_df[date_col])

    crop = pump_precip_df['crop'].values[0]
    site_number = pump_precip_df['Site_ID'].values[0]

    pump_precip_df1 = pump_precip_df.loc[(pump_precip_df[date_col] >= start_date1) &
                                         (pump_precip_df[date_col] <= end_date1)]
    max_pumping1 = pump_precip_df1[pumping_col].max()
    max_precipitation1 = pump_precip_df1[precip_col].max()
    ylim_max1 = max(max_pumping1, max_precipitation1) + 10

    import matplotlib
    matplotlib.rc('xtick', labelsize=7)
    matplotlib.rc('ytick', labelsize=7)

    fig, axs = plt.subplots(2)
    # Subplot 1
    axs[0].plot(pump_precip_df1[date_col], pump_precip_df1[pumping_col], 'ro', alpha=0.2, markersize=3.5)
    axs[0].plot(pump_precip_df1[date_col], pump_precip_df1['7_day_avg_pumping(mm)'], 'r-', alpha=0.5)
    axs[0].set_ylim([0, ylim_max1])
    axs[0].set_ylabel(ax1_ylabel, fontsize=7, color='red')
    axs[0].text(0.02, 0.90, 'crop_type:' + crop + ',' + ' ' + 'Site_number:' + str(site_number),
                transform=axs[0].transAxes, fontsize=6, verticalalignment='top', bbox=dict(facecolor='white',
                                                                                           alpha=0.5))

    ax2 = axs[0].twinx()
    ax2.plot(pump_precip_df1[date_col], pump_precip_df1[precip_col], 'bo', alpha=0.2, markersize=3.5)
    ax2.plot(pump_precip_df1[date_col], pump_precip_df1['7_day_avg_precip(mm)'], 'b-', alpha=0.5)
    ax2.set_ylabel(ax2_ylabel, fontsize=7, color='blue')
    ax2.set_ylim([0, ylim_max1])
    plt.sca(axs[0])
    plt.grid()
    # ax2.plot(pump_precip_df1[date_col], pump_precip_df1[effec_precip_col], color='green', linewidth=0.7,
    # linestyle='--')

    # Subplot 2
    pump_precip_df2 = pump_precip_df.loc[(pump_precip_df[date_col] >= start_date2) &
                                         (pump_precip_df[date_col] <= end_date2)]

    max_pumping2 = pump_precip_df2[pumping_col].max()
    max_precipitation2 = pump_precip_df2[precip_col].max()
    ylim_max2 = max(max_pumping2, max_precipitation2) + 10

    plot1 = axs[1].plot(pump_precip_df2[date_col], pump_precip_df2[pumping_col], 'ro', alpha=0.2, markersize=3.5)
    plot2 = axs[1].plot(pump_precip_df2[date_col], pump_precip_df2['7_day_avg_pumping(mm)'], 'r-', alpha=0.5)
    axs[1].set_ylim([0, ylim_max2])
    axs[1].set_xlabel('Year', fontsize=8)
    axs[1].set_ylabel(ax1_ylabel, fontsize=7, color='red')
    plt.sca(axs[1])
    plt.grid()

    ax2 = axs[1].twinx()
    plot3 = ax2.plot(pump_precip_df2[date_col], pump_precip_df2[precip_col], 'bo', alpha=0.2, markersize=3.5)
    plot4 = ax2.plot(pump_precip_df2[date_col], pump_precip_df2['7_day_avg_precip(mm)'], 'b-', alpha=0.5)
    ax2.set_ylabel(ax2_ylabel, fontsize=7, color='blue')
    ax2.set_ylim([0, ylim_max2])

    plots = plot1 + plot2 + plot3 + plot4
    labels = ['Daily Pumping', '7-day Avg Pumping', 'Daily Rainfall', '7-day Avg Rainfall']
    ax2.legend(plots, labels, loc=0, fontsize=6)
    # ax2.plot(pump_precip_df2[date_col], pump_precip_df2[effec_precip_col], color='green', linewidth=0.7,
    # linestyle='--')

    outdir = pump_precip_csv[:pump_precip_csv.rfind('/') + 1]
    plot_name = outdir + '/' + str(site_number) + '.png'
    plt.savefig(plot_name, dpi=500)


# download_site_data(site_info_excel='../RealtimeMeterNetwork/USGS_Data/StationNotes/MAP20XX_WaterUse_'
#                                        'QuickReferenceSheet_12_08_2020.xlsx',
#                    start_date='2017-01-01', end_date='2021-06-30',
#                    download_dir='../RealtimeMeterNetwork/Site_Daily_Data',
#                    description_download_dir='../RealtimeMeterNetwork/Site_Daily_Data/site_description_data')

# read_save_site_daily_data(outcsv_dir='../RealtimeMeterNetwork/Site_Daily_Data_csv',
#                           site_info_excel='../RealtimeMeterNetwork/USGS_Data/StationNotes/MAP20XX_WaterUse_'
#                                           'QuickReferenceSheet_12_08_2020.xlsx', skiprows=30)

site_datasets = glob(os.path.join('../RealtimeMeterNetwork/Site_Daily_Data_csv', '*.csv'))


# for site in site_datasets:
#     site_with_rainfall_csv = extracting_daily_precipitation_in_pumping_record(
#         pump_csv=site,
#         precipitation_data_dir='../Daily_Precipitation_Data_20160101_20210630',
#         output_dir='../RealtimeMeterNetwork/Site_Rainfall_Data')
#     calculate_effective_rainfall(site_with_rainfall_csv)


# pumping_precip = '../RealtimeMeterNetwork/Site_Rainfall_Data/333149090241801/333149090241801_effective_precip.csv'
# plot_pumping_precipitation(pumping_precip, start_date1='2020-04-01', end_date1='2020-08-31',
#                            start_date2='2021-04-01', end_date2='2021-06-30',date_col='Date',
#                            pumping_col='Pumping(mm)',
#                            precip_col='Observed_precip(mm)', effec_precip_col='Effective_rainfall(mm)',
#                            ax2_ylabel='Precipitation(mm)')

def create_dataframe_with_precipitation_lag(output_csv,
                                            pump_precip_data_dir='../RealtimeMeterNetwork/Site_Rainfall_Data',
                                            precipitation_col='Observed_precip(mm)', search_by='*with_obs_precip*.csv'):
    """
    Create a dataframe (csv) with 1, 2, 3,... day precipitation lag data.

    :param output_csv: Output csv filepath.
    :param pump_precip_data_dir: Directory of pumping station data (pump data must have both pumping and precipitation
                                 data)
    :param precipitation_col: Precipitation column name in input csv.
    :param search_by: Search criteria for selecting pump_precipitation datasets.

    :return: A csv with precipitation lag data along with pumping and precipitation data for all pump stations.
    """
    datasets_pump_precip = glob(os.path.join(pump_precip_data_dir, '*', search_by))
    final_df = pd.DataFrame()
    for data in datasets_pump_precip:
        df = pd.read_csv(data)
        precipitation = df[precipitation_col].to_list()
        precipitation.insert(0, np.nan)
        df['precip_1_day_lag'] = precipitation[:-1]
        precipitation.insert(0, np.nan)
        df['precip_2_day_lag'] = precipitation[:-2]
        precipitation.insert(0, np.nan)
        df['precip_3_day_lag'] = precipitation[:-3]
        precipitation.insert(0, np.nan)
        df['precip_4_day_lag'] = precipitation[:-4]
        precipitation.insert(0, np.nan)
        df['precip_5_day_lag'] = precipitation[:-5]
        precipitation.insert(0, np.nan)
        df['precip_6_day_lag'] = precipitation[:-6]
        precipitation.insert(0, np.nan)
        df['precip_7_day_lag'] = precipitation[:-7]
        df = df.dropna(subset=['precip_1_day_lag', 'precip_2_day_lag', 'precip_3_day_lag', 'precip_4_day_lag',
                               'precip_5_day_lag', 'precip_6_day_lag', 'precip_7_day_lag'])
        final_df = final_df.append(df, ignore_index=True)
    final_df.to_csv(output_csv, index=False)


# create_dataframe_with_precipitation_lag('../RealtimeMeterNetwork/All_pumps_precip_ML.csv')

def create_dataframe_with_precipitation_average(output_csv,
                                                pump_precip_data_dir='../RealtimeMeterNetwork/Site_Rainfall_Data',
                                                precipitation_col='Observed_precip(mm)',
                                                search_by='*with_obs_precip*.csv'):
    """
    Create a dataframe (csv) with 1, 2, 3,... day precipitation average data.

    :param output_csv: Output csv filepath.
    :param pump_precip_data_dir: Directory of pumping station data (pump data must have both pumping and precipitation
                                 data)
    :param precipitation_col: Precipitation column name in input csv.
    :param search_by: Search criteria for selecting pump_precipitation datasets.

    :return: A csv with precipitation lag data along with pumping and precipitation data for all pump stations.
    """
    datasets_pump_precip = glob(os.path.join(pump_precip_data_dir, '*', search_by))
    final_df = pd.DataFrame()
    for data in datasets_pump_precip:
        df = pd.read_csv(data)
        precipitation = df[precipitation_col].to_list()
        avg_2_day_precip = []
        avg_3_day_precip = []
        avg_4_day_precip = []
        avg_5_day_precip = []
        avg_6_day_precip = []
        avg_7_day_precip = []
        for i in range(7, len(precipitation)):
            avg_2_day = mean(precipitation[i - 1:i + 1])
            avg_2_day_precip.append(avg_2_day)
            avg_3_day = mean(precipitation[i - 2:i + 1])
            avg_3_day_precip.append(avg_3_day)
            avg_4_day = mean(precipitation[i - 3:i + 1])
            avg_4_day_precip.append(avg_4_day)
            avg_5_day = mean(precipitation[i - 4:i + 1])
            avg_5_day_precip.append(avg_5_day)
            avg_6_day = mean(precipitation[i - 5:i + 1])
            avg_6_day_precip.append(avg_6_day)
            avg_7_day = mean(precipitation[i - 6:i + 1])
            avg_7_day_precip.append(avg_7_day)
        df = df[7:]
        df['avg_2_day_precip'] = avg_2_day_precip
        df['avg_3_day_precip'] = avg_3_day_precip
        df['avg_4_day_precip'] = avg_4_day_precip
        df['avg_5_day_precip'] = avg_5_day_precip
        df['avg_6_day_precip'] = avg_6_day_precip
        df['avg_7_day_precip'] = avg_7_day_precip

        final_df = final_df.append(df, ignore_index=True)
    final_df.to_csv(output_csv, index=False)


# create_dataframe_with_precipitation_average('../RealtimeMeterNetwork/All_pumps_precip_Avg_ML.csv')

# dataframe = pd.read_csv('../RealtimeMeterNetwork/All_pumps_precip_ML.csv')
# dataframe = dataframe.dropna(axis=0)
# cdl_dict = {
#     'corn': 1,
#     'cotton': 2,
#     'rice': 3,
#     'soybeans': 5,
#     'catfish': 92,
#     'other': 0
# }
# df_model = dataframe[['precip_1_day_lag', 'precip_2_day_lag', 'precip_3_day_lag', 'precip_4_day_lag',
#                       'precip_5_day_lag', 'precip_6_day_lag', 'precip_7_day_lag', '7_day_avg_precip(mm)',
#                       'crop', 'Pumping(mm)']].copy()
# crop_list = df_model['crop']
# crop_number = [cdl_dict.get(crop, np.nan) for crop in crop_list]
# df_model = df_model.drop(columns={'crop'})
# df_model['crop_number'] = crop_number
# df_corn = df_model[df_model['crop_number'] == 1]
# df_cotton = df_model[df_model['crop_number'] == 2]
# df_rice = df_model[df_model['crop_number'] == 3]
# df_soybeans = df_model[df_model['crop_number'] == 5]
# df_catfish = df_model[df_model['crop_number'] == 92]
#
# # whole
# x = df_model.loc[:, df_model.columns != 'Pumping(mm)']
# y = df_model['Pumping(mm)']
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3, shuffle=True)
# rf_model = RandomForestRegressor(n_estimators=400, max_depth=6)
# rf_model.fit(x_train, y_train)
# y_pred = rf_model.predict(x_test)
# print('MSE :', metrics.mean_squared_error(y_test, y_pred))
# print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('R2 :', metrics.r2_score(y_test, y_pred))
#
# plt.plot(y_test, y_pred, 'bo', alpha=0.5)
# plt.xlabel('Pumping_actual(mm)')
# plt.ylabel('Pumping_predicted(mm)')
# plt.show()
#
# # corn
# x = df_corn.loc[:, df_corn.columns != 'Pumping(mm)']
# y = df_corn['Pumping(mm)']
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3, shuffle=True)
# rf_model = RandomForestRegressor(n_estimators=400, max_depth=6)
# rf_model.fit(x_train, y_train)
# y_pred = rf_model.predict(x_test)
# print('MSE :', metrics.mean_squared_error(y_test, y_pred))
# print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('R2 :', metrics.r2_score(y_test, y_pred))
#
# plt.plot(y_test, y_pred, 'bo', alpha=0.5)
# plt.xlabel('Pumping_actual(mm)')
# plt.ylabel('Pumping_predicted(mm)')
# plt.show()
#
# # cotton
# x = df_cotton.loc[:, df_cotton.columns != 'Pumping(mm)']
# y = df_cotton['Pumping(mm)']
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3, shuffle=True)
# rf_model = RandomForestRegressor(n_estimators=400, max_depth=6)
# rf_model.fit(x_train, y_train)
# y_pred = rf_model.predict(x_test)
# print('MSE :', metrics.mean_squared_error(y_test, y_pred))
# print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('R2 :', metrics.r2_score(y_test, y_pred))
#
# plt.plot(y_test, y_pred, 'bo', alpha=0.5)
# plt.xlabel('Pumping_actual(mm)')
# plt.ylabel('Pumping_predicted(mm)')
# plt.show()
#
# # rice
# x = df_rice.loc[:, df_rice.columns != 'Pumping(mm)']
# y = df_rice['Pumping(mm)']
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3, shuffle=True)
# rf_model = RandomForestRegressor(n_estimators=400, max_depth=6)
# rf_model.fit(x_train, y_train)
# y_pred = rf_model.predict(x_test)
# print('MSE :', metrics.mean_squared_error(y_test, y_pred))
# print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('R2 :', metrics.r2_score(y_test, y_pred))
#
# plt.plot(y_test, y_pred, 'bo', alpha=0.5)
# plt.xlabel('Pumping_actual(mm)')
# plt.ylabel('Pumping_predicted(mm)')
# plt.show()
#
#
# # soybeans
# x = df_soybeans.loc[:, df_soybeans.columns != 'Pumping(mm)']
# y = df_soybeans['Pumping(mm)']
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3, shuffle=True)
# rf_model = RandomForestRegressor(n_estimators=400, max_depth=6)
# rf_model.fit(x_train, y_train)
# y_pred = rf_model.predict(x_test)
# print('MSE :', metrics.mean_squared_error(y_test, y_pred))
# print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('R2 :', metrics.r2_score(y_test, y_pred))
#
# plt.plot(y_test, y_pred, 'bo', alpha=0.5)
# plt.xlabel('Pumping_actual(mm)')
# plt.ylabel('Pumping_predicted(mm)')
# plt.show()
#
# # catfish
# x = df_catfish.loc[:, df_catfish.columns != 'Pumping(mm)']
# y = df_catfish['Pumping(mm)']
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3, shuffle=True)
# rf_model = RandomForestRegressor(n_estimators=400, max_depth=6)
# rf_model.fit(x_train, y_train)
# y_pred = rf_model.predict(x_test)
# print('MSE :', metrics.mean_squared_error(y_test, y_pred))
# print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('R2 :', metrics.r2_score(y_test, y_pred))
#
# plt.plot(y_test, y_pred, 'bo', alpha=0.5)
# plt.xlabel('Pumping_actual(mm)')
# plt.ylabel('Pumping_predicted(mm)')
# plt.show()



# dataframe = pd.read_csv('../RealtimeMeterNetwork/All_pumps_precip_Avg_ML.csv')
# dataframe = dataframe.dropna(axis=0)
# cdl_dict = {
#     'corn': 1,
#     'cotton': 2,
#     'rice': 3,
#     'soybeans': 5,
#     'catfish': 92,
#     'other': 0
# }
# df_model = dataframe[['avg_2_day_precip', 'avg_3_day_precip', 'avg_4_day_precip', 'avg_5_day_precip',
#                       'avg_6_day_precip', 'avg_7_day_precip', 'crop', 'Pumping(mm)']].copy()
# crop_list = df_model['crop']
# crop_number = [cdl_dict.get(crop, np.nan) for crop in crop_list]
# df_model = df_model.drop(columns={'crop'})
# df_model['crop_number'] = crop_number
# df_corn = df_model[df_model['crop_number'] == 1]
# df_cotton = df_model[df_model['crop_number'] == 2]
# df_rice = df_model[df_model['crop_number'] == 3]
# df_soybeans = df_model[df_model['crop_number'] == 5]
# df_catfish = df_model[df_model['crop_number'] == 92]
#
# # whole
# x = df_model.loc[:, df_model.columns != 'Pumping(mm)']
# y = df_model['Pumping(mm)']
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3, shuffle=True)
# rf_model = RandomForestRegressor(n_estimators=400, max_depth=6)
# rf_model.fit(x_train, y_train)
# y_pred = rf_model.predict(x_test)
# print('MSE :', metrics.mean_squared_error(y_test, y_pred))
# print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('R2 :', metrics.r2_score(y_test, y_pred))
#
# plt.plot(y_test, y_pred, 'bo', alpha=0.5)
# plt.xlabel('Pumping_actual(mm)')
# plt.ylabel('Pumping_predicted(mm)')
# plt.show()