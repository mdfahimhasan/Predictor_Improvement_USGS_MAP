import os
from glob import glob
import rasterio
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


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

def download_site_data(site_info_excel='./RealtimeMeterNetwork/USGS_Data/StationNotes/MAP20XX_WaterUse_'
                                       'QuickReferenceSheet_12_08_2020.xlsx',
                       start_date='2017-01-01', end_date='2021-07-31',
                       download_dir='./RealtimeMeterNetwork/Site_Daily_Data',
                       description_download_dir='./RealtimeMeterNetwork/Site_Daily_Data/site_description_data'):
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
#     new_df = pd.DataFrame()
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
#         new_df = new_df.append(year_df)
#
#     new_df['Pumping(mm)'] = new_df['Pumping(in)']*25.4
#     new_df['Permit Number'] = permit_number
#
#     csv_name = txtfilepath[txtfilepath.rfind('/') + 1:txtfilepath.rfind('.')] + '.csv'
#     new_df.to_csv(os.path.join(outcsv_dir, csv_name), index=False)


# read_save_site_daily_data(txtfilepath='H:/USGS_MAP/Fahim/RealtimeMeterNetwork/Site_Daily_Data/332245090320901.txt',
#                           outcsv_dir='H:/USGS_MAP/Fahim/RealtimeMeterNetwork/Site_Daily_Data',
#                           lat_deg=33, lat_min=22, lat_sec=45, lon_deg=90, lon_min=32, lon_sec=9,
#                           permit_number='MS-GW-44991', skiprows=27)


def read_save_site_daily_data(outcsv_dir,
                              site_info_excel='./RealtimeMeterNetwork/USGS_Data/StationNotes/MAP20XX_WaterUse_'
                                              'QuickReferenceSheet_12_08_2020.xlsx', skiprows=30,
                              site_daily_data_dir = './RealtimeMeterNetwork/Site_Daily_Data',
                              site_description_dir='./RealtimeMeterNetwork/Site_Daily_Data/site_description_data'):
    """
    Reading and Saving .txt daily data downloaded from USGS Real Time Server as csv file
    (https://waterdata.usgs.gov/nwis/inventory?search_criteria=search_site_no&submitted_form=introduction).

    :param txtfilepath: File path of the daily data. Must be in .txt format.
    :param outcsv_dir: Output directory path to save the daily data as csv file.
    :param site_info_excel: Realtime site information excel.
    :param skiprows: Number of rows to skip while reading the file with pandas. Default set to 28 line.

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

        site_number = site[site.rfind(os.sep)+1:site.rfind('.')]
        site_description_txt = os.path.join(site_description_dir, str(site_number) + '_site_description.txt')
        description_df = pd.read_csv(site_description_txt, sep='\t', index_col=False, skiprows=24)
        description_df = description_df.iloc[:, 0:2]
        description_df.columns = ['Lat_WGS84', 'Lon_WGS84']
        Latitude = description_df['Lat_WGS84'][0]
        Longitude = description_df['Lon_WGS84'][0]

        df['Latitude_WGS84'] = Latitude
        df['Longitude_WGS84'] = Longitude
        site_id = df['Site_ID'][0]

        site_info_df = pd.read_excel(site_info_excel, sheet_name='2020')
        combined_acre = site_info_df[site_info_df['Site_number'] == site_id]['combined acres'].values[0]
        crop = site_info_df[site_info_df['Site_number'] == site_id]['crop'].values[0]

        # creating a column 'Pumping(in) in the database
        df['Pumping(in)'] = df['Pumping(Acre-ft)'] * 12 / combined_acre
        df['Pumping(mm)'] = df['Pumping(in)'] * 25.4
        df['Acre'] = combined_acre
        df['crop'] = crop

        csv_name = site[site.rfind(os.sep) + 1:site.rfind('.')] + '.csv'
        df.to_csv(os.path.join(outcsv_dir, csv_name), index=False)


def extracting_daily_precipitation_for_pump_record(pump_csv, precipitation_data_dir,
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

    site_number = pump_csv[pump_csv.rfind('/')+1:pump_csv.rfind('.')]
    final_output_dir = output_dir + '/' + site_number

    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)

    output_csv = final_output_dir + '/' + site_number + '_with_obs_precip.csv'
    pump_df.to_csv(output_csv, index=False)

    return output_csv


def find_effective_rainfall(rainfall_value, percent=90):
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


# Extracting daily rainfall data for pump coordinates
def extract_daily_precip_for_pump(pumping_data_with_precip_csv,
                                  precip_data_dir='Daily_Precipitation_Data_20160101_20210630'):

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


# extract_daily_precip_for_pump(pumping_data_with_precip_csv='./RealtimeMeterNetwork/Site_Daily_Data/332245090320901/332245090320901.csv')
def calculate_effective_rainfall(pumping_data_with_precip_csv, daily_rainfall_csv=None,
                                 precip_data_dir='Daily_Precipitation_Data_20160101_20210630',
                                 extract_rainfall_value=False):
    if extract_rainfall_value:
        daily_rainfall_csv = extract_daily_precip_for_pump(pumping_data_with_precip_csv, precip_data_dir)
        daily_precip_df = pd.read_csv(daily_rainfall_csv)
    else:
        daily_precip_df = pd.read_csv(daily_rainfall_csv)

    daily_precip_df['Date'] = pd.to_datetime(daily_precip_df['Date'])
    daily_precip_df['Year'] = pd.DatetimeIndex(daily_precip_df['Date']).year
    yearlist = daily_precip_df['Year'].unique().tolist()
    monthlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    mean_monthly_precipitation = {}  # dictionary for mean monthly precipitation for each month for all the years
    monthly_precip_dict = {}  # created for storing monthly total precipitaion for each year
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

    monthly_effective_rainfall_dict = {}  # dictionary for stroing effective monthly rainfall for each month
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
    pump_rainfall_df['Effective_rainfall(in)'] = pump_rainfall_df['Effective_rainfall(mm)']/25.4

    csv_name = pumping_data_with_precip_csv[
               pumping_data_with_precip_csv.rfind('/') + 1:pumping_data_with_precip_csv.rfind('.')]
    station_id = csv_name.split('_')[0]
    outdir = pumping_data_with_precip_csv[:pumping_data_with_precip_csv.rfind('/')]
    output_csv = os.path.join(outdir, station_id + '_effective_precip.csv')
    pump_rainfall_df.to_csv(output_csv, index=False)


def plot_pumping_precipitation(pump_precip_csv, start_date1, end_date1, start_date2, end_date2, date_col='Date',
                               pumping_col='Pumping(mm)', precip_col='Observed_precip(mm)',
                               effec_precip_col='Effective_rainfall(mm)',
                               ax1_ylabel='Pumping (mm)', ax2_ylabel='Precipitation (mm)'):
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

    pump_precip_df2 = pump_precip_df.loc[(pump_precip_df[date_col] >= start_date2) &
                                         (pump_precip_df[date_col] <= end_date2)]

    max_pumping2 = pump_precip_df2[pumping_col].max()
    max_precipitation2 = pump_precip_df2[precip_col].max()
    ylim_max2 = max(max_pumping2, max_precipitation2) + 10

    fig, axs = plt.subplots(2)
    # Subplot 1
    axs[0].plot(pump_precip_df1[date_col], pump_precip_df1[pumping_col], color='red', marker='o', linewidth=0.5,
                markersize=1.5)
    axs[0].set_ylim([0, ylim_max1])
    axs[0].set_ylabel(ax1_ylabel, fontsize=9, color='red')
    axs[0].text(0.70, 0.90, 'crop_type:' + crop, transform=axs[0].transAxes, fontsize=7, verticalalignment='top')

    ax2 = axs[0].twinx()
    ax2.plot(pump_precip_df1[date_col], pump_precip_df1[precip_col], color='blue', marker='o', linewidth=0.5,
             markersize=1.5)
    ax2.set_ylabel(ax2_ylabel, fontsize=9, color='blue')
    ax2.set_ylim([0, ylim_max1])
    ax2.plot(pump_precip_df1[date_col], pump_precip_df1[effec_precip_col], color='green', linewidth=0.7, linestyle='--')

    # Subplot 2
    axs[1].plot(pump_precip_df2[date_col], pump_precip_df2[pumping_col], color='red', marker='o', linewidth=0.5,
                markersize=1.5)
    axs[1].set_ylim([0, ylim_max2])
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel(ax1_ylabel, fontsize=9, color='red')
    axs[1].text(0.65, 0.95, 'Site_number:' + str(site_number), transform=axs[1].transAxes, fontsize=7,
                verticalalignment='top')

    ax2 = axs[1].twinx()
    ax2.plot(pump_precip_df2[date_col], pump_precip_df2[precip_col], color='blue', marker='o', linewidth=0.5,
             markersize=1.5)
    ax2.set_ylabel(ax2_ylabel, fontsize=9, color='blue')
    ax2.set_ylim([0, ylim_max2])
    ax2.plot(pump_precip_df2[date_col], pump_precip_df2[effec_precip_col], color='green', linewidth=0.7, linestyle='--')

    outdir = pump_precip_csv[:pump_precip_csv.rfind('/')+1]
    plot_name = outdir + '/' + str(site_number) + '.png'
    plt.savefig(plot_name)


# download_site_data(site_info_excel='./RealtimeMeterNetwork/USGS_Data/StationNotes/MAP20XX_WaterUse_'
#                                        'QuickReferenceSheet_12_08_2020.xlsx',
#                    start_date='2017-01-01', end_date='2021-06-30',
#                    download_dir='./RealtimeMeterNetwork/Site_Daily_Data',
#                    description_download_dir='./RealtimeMeterNetwork/Site_Daily_Data/site_description_data')

# read_save_site_daily_data(outcsv_dir='./RealtimeMeterNetwork/Site_Daily_Data_csv',
#                           site_info_excel='./RealtimeMeterNetwork/USGS_Data/StationNotes/MAP20XX_WaterUse_'
#                                           'QuickReferenceSheet_12_08_2020.xlsx', skiprows=30)


# site_with_rainfall_csv = extracting_daily_precipitation_for_pump_record(
#     pump_csv='./RealtimeMeterNetwork/Site_Daily_Data_csv/352211090540301.csv',
#     precipitation_data_dir='./Daily_Precipitation_Data_20160101_20210630',
#     output_dir='./RealtimeMeterNetwork/Site_Rainfall_Data')
#
# calculate_effective_rainfall(site_with_rainfall_csv,
#                              extract_rainfall_value=True)


# pumping_precip = './RealtimeMeterNetwork/Site_Rainfall_Data/334702090505901/334702090505901_effective_precip.csv'
# plot_pumping_precipitation(pumping_precip, start_date1='2018-05-25', end_date1='2018-11-30',
#                            start_date2='2019-05-25', end_date2='2019-11-30',date_col='Date', pumping_col='Pumping(mm)',
#                            precip_col='Observed_precip(mm)', effec_precip_col='Effective_rainfall(mm)',
#                            ax2_ylabel='Precipitation(mm)')
