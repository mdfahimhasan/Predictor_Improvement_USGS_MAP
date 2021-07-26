import os
import ee
import gdal
import requests
import pandas as pd
from glob import glob
import geopandas as gpd
import zipfile
import datetime
import numpy as np
import rasterio as rio
from rasterio.merge import merge

NO_DATA_VALUE = -9999
referenceraster1 = './Sentinel_2 data Download/CatfishPond/refraster_Catfish_AOI1.tif'
referenceraster2 = './Sentinel_2 data Download/CatfishPond/refraster_Catfish_AOI2.tif'


def read_raster_arr_object(input_raster, band=1, raster_object=False, get_file=True, change_dtype=True):
    """
    read raster as raster object and array. If raster_object=True get only the raster array

    Parameters
    ----------
    input_raster : Input raster file path
    band : Selected band to read (Default 1)
    raster_object : Set true if raster_file is a rasterio object
    get_file : Get both rasterio object and raster array file if set to True
    change_dtype : Change raster data type to float if true
    ----------
    Returns  Raster numpy array and rasterio object file (rasterio_obj=False and get_file=True)
    """
    if not raster_object:
        raster_file = rio.open(input_raster)
    else:
        get_file = False

    raster_arr = raster_file.read(band)
    if change_dtype:
        raster_arr = raster_arr.astype(np.float32)
        if raster_file.nodata:
            raster_arr[np.isclose(raster_arr, raster_file.nodata)] = np.nan

    if get_file:
        return raster_arr, raster_file
    return raster_arr


def write_raster(raster_arr, raster_file, transform, outfile_path, no_data_value=NO_DATA_VALUE,
                 ref_file=None):
    """
    Write raster file in GeoTIFF format

    Parameters
    ----------
    raster_arr: Raster array data to be written
    raster_file: Original rasterio raster file containing geo-coordinates
    transform: Affine transformation matrix
    outfile_path: Outfile file path with txtfilepath
    no_data_value: No data value for raster (default float32 type is considered)
    ref_file: Write output raster considering parameters from reference raster file
    ----------
    Returns  None
    """
    if ref_file:
        raster_file = rio.open(ref_file)
        transform = raster_file.transform
    with rio.open(
            outfile_path,
            'w',
            driver='GTiff',
            height=raster_arr.shape[0],
            width=raster_arr.shape[1],
            dtype=raster_arr.dtype,
            crs=raster_file.crs,
            transform=transform,
            count=raster_file.count,
            nodata=no_data_value
    ) as dst:
        dst.write(raster_arr, raster_file.count)

    return outfile_path


def mosaic_rasters(input_dir, output_dir, raster_name, search_by="*.tif", ref_raster=referenceraster1,
                   resolution=10, no_data=NO_DATA_VALUE):
    """
    Mosaics multiple rasters into a single raster (rasters have to be in the same directory).

    Parameters:
    input_dir : Input rasters directory.
    output_dir : Outpur raster directory.
    raster_name : Outpur raster name.
    ref_raster : Reference raster with filepath.
    no_data : No data value. Default -9999.
    resolution: Resolution of the output raster.

    Returns: Mosaiced Raster.
    """
    input_rasters = glob(os.path.join(input_dir, search_by))

    raster_list = []
    for raster in input_rasters:
        arr, file = read_raster_arr_object(raster)
        raster_list.append(file)

    ref_arr, ref_file = read_raster_arr_object(ref_raster)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_raster = os.path.join(output_dir, raster_name)
    out_vrt = os.path.join(output_dir, 'merged.vrt')

    vrt = gdal.BuildVRT(out_vrt, input_rasters)
    gdal.Translate(out_raster, vrt, xRes=resolution, yRes=-resolution, format='GTiff', outputSRS=ref_file.crs,
                   noData=no_data)
    vrt = None


def extract_data(zip_dir, out_dir, searchby="*.zip", rename_file=True):
    """
    Extract zipped data
    Parameters
    ----------
    zip_dir : File Location
    out_dir : File Location where data will be extracted
    searchby : Keyword for searching files, default is "*.zip".
    rename_file : True if file rename is required while extracting
    """
    print('Extracting zip files.....')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for zip_file in glob(os.path.join(zip_dir, searchby)):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            if rename_file:
                zip_key = zip_file[zip_file.rfind(os.sep) + 1:zip_file.rfind(".")]
                zip_info = zip_ref.infolist()[0]
                zip_info.filename = zip_key + '.tif'
                zip_ref.extract(zip_info, path=out_dir)
            else:
                zip_ref.extractall(path=out_dir)


def cloudmaskS2sr(image):
    """
    Function to mask clouds based on the pixel_qa band of Sentinel-2 SR data. Used in combination with Sentinel-2
    GEE download function.

    param : {ee.Image} image input Sentinel-2 SR image
    return : {ee.Image} cloudmasked Sentinel-2 image
    """
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = (1 << 10)
    cirrusBitMask = (1 << 11)
    # Get the pixel QA band.
    qa = image.select('QA60')
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
        .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask)


def download_sentinel2_data(yearlist, start_month, end_month, output_dir, shapecsv,
                            interval=30, gee_scale=20, bandname='B8',
                            imagecollection='COPERNICUS/S2_SR', factor=0.0001, refraster=referenceraster1):
    """
    Download data from Sentinel-2, with cloudmask applied, from Google Earth Engine for range of years.

    ***The code works best for small spatial and temporal scale data (Otherwise takes a lot of time).***

    Parameters:
    yearlist :  Year for which data will be downloaded, i.e., [2010,2020].
    start_month : Start month of data.
    end_month : End month of data.
    output_dir : Location to downloaded data.
    shapecsv : Location of input csv with grid coordinates.
    interval : Download data interval. Sentinel-2 revisit period is 5 days. Default set to 10 days.
    gee_scale : Download Scale in meter. Defaults to 10m.
    bandname : Band to download from Google earth engine.
    imagecollection : Imagecollection name.
    factor : Scale/Factor mentioned in GEE band (if needed) to multiply with the band.
    refraster : Reference raster for merging the downloaded datasets. Set to referenceraster1 for AOI1.
                Set to referenceraster2 for AOI2.
    """
    # Initialize
    ee.Initialize()
    data_download = ee.ImageCollection(imagecollection)

    # Creating Output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Date Range Creation
    for year in yearlist:
        # Creating a Datelist to creating Daterange
        start_day = datetime.date(year, start_month, 1)
        if end_month in [1, 3, 5, 6, 8, 10, 12]:
            end_day = datetime.date(year, end_month, 31)
        elif end_month == 2:
            end_day = datetime.date(year, end_month, 28)
        else:
            end_day = datetime.date(year, end_month, 30)

        num_intervals = int((end_day - start_day).days/interval)
        datelist = []
        for k in range(0, num_intervals + 1):
            if k == 0:
                start_day = start_day
                date_str = start_day.strftime('%Y-%m-%d')
                datelist.append(date_str)
            else:
                start_date_index = k - 1
                split = datelist[start_date_index].split('-')
                start_day = datetime.date(int(split[0]), int(split[1]),
                                          int(split[2]))  # splited string date to create datetime object again

                new_date = start_day + datetime.timedelta(days=interval)
                date_str = new_date.strftime('%Y-%m-%d')
                datelist.append(date_str)

        if end_month == 2:
            end_date= datetime.date(year, 2, 28)
            end_str= end_date.strftime('%Y-%m-%d')
            datelist.append(end_str)
        print(datelist)

        # Selecting data for each pair of dates from the datelist
        for i in range(0, len(datelist)):
            if i == len(datelist)-1:
                pass
            else:
                start_date = ee.Date(datelist[i])
                end_date = ee.Date(datelist[i+1])

                # Collecting Data
                cloudmasked = data_download.filterDate(start_date, end_date).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15)).\
                    map(cloudmaskS2sr).select(bandname).median().multiply(factor).toFloat()

                # .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))  between filterDate and map(cloudmaskS2sr)

                coords_df = pd.read_csv(shapecsv)
                for index, row in coords_df.iterrows():
                    # Define Extent
                    minx = row['minx']
                    miny = row['miny']
                    maxx = row['maxx']
                    maxy = row['maxy']
                    gee_extent = ee.Geometry.Rectangle((minx, miny, maxx, maxy))

                    filename = row['shape']+datelist[i]+'_'+datelist[i+1]
                    # Download URL
                    data_url = cloudmasked.getDownloadURL({'name': filename,
                                                           'crs': "EPSG:6344",  # download projection set to NAD83
                                                           'scale': gee_scale,
                                                           'region': gee_extent})
                    # Dowloading data
                    outdir = os.path.join(output_dir, datelist[i] + '_' + datelist[i + 1])
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)

                    local_file_name = os.path.join(outdir, filename + '.zip')
                    print('Downloading', local_file_name, '.....')
                    r = requests.get(data_url, allow_redirects=True)
                    open(local_file_name, 'wb').write(r.content)

                    if index == coords_df.index[-1]:
                        extract_data(zip_dir=outdir, out_dir=outdir, rename_file=False)

                        mosaiced_dir = os.path.join(outdir, 'merged_data')
                        if not os.path.exists(mosaiced_dir):
                            os.makedirs(mosaiced_dir)

                        mosaic_name = datelist[i] + '_' + datelist[i+1] + '.tif'
                        print('Merging datasets to..', mosaic_name)
                        mosaic_rasters(input_dir=outdir, output_dir=mosaiced_dir, raster_name=mosaic_name,
                                       ref_raster=refraster, search_by='*.tif', resolution=10,
                                       no_data=NO_DATA_VALUE)
    print('All Data Downloaded')


csv = './Sentinel_2 data Download/Realtime_FieldBoundaries/Realtime_fields.csv'

# B4
# CatfishAOI3
outdir = './Downloadeddata_B4/CatfishAOI3'

# June 2019- September 2019
download_sentinel2_data(yearlist=[2019], start_month=6, end_month=9, output_dir=outdir, shapecsv=csv,
                            gee_scale=10, bandname='B4', factor=0.0001, interval=30,
                            imagecollection='COPERNICUS/S2_SR', refraster=referenceraster1)


# # B2
# # CatfishAOI3
# outdir = './Sentinel_2 data Download/Downloadeddata_B2/CatfishAOI3'
#
# # June 2019- September 2019
# download_sentinel2_data(yearlist=[2019], start_month=6, end_month=9, output_dir=outdir, shapecsv=csv,
#                             gee_scale=10, bandname='B2', factor=0.0001, interval=30,
#                             imagecollection='COPERNICUS/S2_SR', refraster=referenceraster1)


# # B3
# outdir = './Sentinel_2 data Download/Downloadeddata_B3/CatfishAOI3'
#
# # June 2019- September 2019
# download_sentinel2_data(yearlist=[2019], start_month=6, end_month=9, output_dir=outdir, shapecsv=csv,
#                             gee_scale=10, bandname='B3', factor=0.0001, interval=30,
#                             imagecollection='COPERNICUS/S2_SR', refraster=referenceraster1)
#
# # B8
# outdir = r'./Sentinel_2 data Download/Downloadeddata_B8/CatfishAOI3'
#
# # June 2019- September 2019
# download_sentinel2_data(yearlist=[2019], start_month=6, end_month=9, output_dir=outdir, shapecsv=csv,
#                             gee_scale=10, bandname='B8', factor=0.0001, interval=30,
#                             imagecollection='COPERNICUS/S2_SR', refraster=referenceraster1)
