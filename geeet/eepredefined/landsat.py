"""
Optional module to define some useful ee functions related to Landsat images processing. 
"""
import ee, datetime, warnings
import geeet
from typing import Union, Any, Dict, List, Literal, Callable

def scale_SR(img:ee.Image)->ee.Image:
    """Scales the optical and thermal bands (SR_B.* and ST_B.*)
    """
    # Scaling factors for Collection 02, level 2:
    opticalBands = img.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = img.select('ST_B.*').multiply(0.00341802).add(149.0)

    return (img.addBands(opticalBands, overwrite=True)
        .addBands(thermalBands, overwrite=True)
    )

def cloud_mask(img: ee.Image, name:str ="cloud_cover") -> ee.Image:
    """Adds a "cloud_cover" band. 

    Requires a "QA_PIXEL" and "QA_RADSAT" band. 

    - 1: cloud
    - 0: not cloud
    - masked: L7 stripes or outside of the image. 

    See also: update_cloud_mask

    """
    qa_mask = img.select('QA_PIXEL').bitwiseAnd(int('11111',2)).neq(0)
    saturation_mask = img.select('QA_RADSAT').neq(0)
    cloud_mask=qa_mask.max(saturation_mask).rename(name) 
    return img.addBands(cloud_mask)


def cfmask(bandNames:list)->Callable:
    def apply_mask(img:ee.Image)->ee.Image:
        """Returns img with the cloud mask used to 
        update the mask on selected bands (bandNames)
        Requires bands: cloud_cover
        """
        cloud_mask = ee.Image(1).subtract(img.select("cloud_cover"))
        imgBands = (img.select(bandNames)
        .updateMask(cloud_mask)
        )
        return img.addBands(imgBands, overwrite=True)

    return apply_mask


def set_index(img: ee.Image) -> ee.Image:
    return img.set({'LANDSAT_INDEX':img.get('system:index'),
        'LANDSAT_FOOTPRINT': img.get('system:footprint')})


def add_ndvi(img:ee.Image)->ee.Image:
    """Adds NDVI to a Landsat 7*, 8, or 9 image. 
    *Bands are renamed in place to match Landsat 8/9, but not returned. 
    bands: nir, red
    """
    band_names = ["SR_B5", "SR_B4"] 
    bands_l7 = ["SR_B4", "SR_3"]
    spacecraft_id = ee.String(img.get('SPACECRAFT_ID'))
    sel_bands = ee.Algorithms.If(spacecraft_id.equals('LANDSAT_7'),bands_l7, band_names)
    ndvi = (img.select(sel_bands, band_names)
            .normalizedDifference(band_names)
            .rename('NDVI'))
    return img.addBands(ndvi.clamp(-1,1))


def albedo_tasumi(img:ee.Image):
    """
    Tasumi et al (2008) albedo parameterization 

    Reference
    ---
    [Tasumi et al. (2008)](https://doi.org/10.1061/(ASCE)1084-0699(2008)13:2(51))
    [Ke et al. (2016)](https://doi.org/10.3390/rs8030215)

    """
    oli = '0.130*b("SR_B1") + 0.115*b("SR_B2") + 0.143*b("SR_B3") + 0.180*b("SR_B4") + 0.281*b("SR_B5") + 0.108*b("SR_B6") + 0.042*b("SR_B7")'
    etm = '0.254*b("SR_B1") + 0.149*b("SR_B2") + 0.147*b("SR_B3") + 0.311*b("SR_B4") + 0.103*b("SR_B5")                    + 0.036*b("SR_B7")'
    spacecraft_id = ee.String(img.get('SPACECRAFT_ID'))
    expr = ee.Algorithms.If(spacecraft_id.equals('LANDSAT_7'),etm, oli)
    albedo = img.expression(expr).rename('albedo')
    return albedo


def add_albedo_tasumi(img:ee.Image)->ee.Image:
    """Adds albedo based on Tasumi et al. 2008 to a Landsat 7, 8, or 9 image.
    """
    alb = albedo_tasumi(img)
    return img.addBands(alb.clamp(0,1))


def albedo_liang(img:ee.Image, 
coefs=[0.356, 0.130, 0.373, 0.085, 0.072, -0.0018],
bands = ['SR_B2', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])->ee.Image:
    """
    Liang (2001) albedo parameterization 
    for Landsat data: 
    0.356*blue + 0.130*red + 0.373*nir + 0.085*swir + 0.072*swir2 - 0.0018

    Optional inputs (defaults to Landsat-8 coefficients and band names*):
    coefs: 6 coefficients for a Landsat sensor
    bands: names of the bands in the Landsat ee.Image
        (corresponding to the blue, red, nir, swir, and swir2 bands)

    *for collection 02, level 2 (SR) product. 

    see also albedo_liang_vis, albedo_liang_nir

    Reference
    ---
    https://www.sciencedirect.com/science/article/pii/S0034425700002054

    """
    bblue = img.select(bands[0]).multiply(coefs[0])
    bred = img.select(bands[1]).multiply(coefs[1])
    bnir = img.select(bands[2]).multiply(coefs[2])
    bswir = img.select(bands[3]).multiply(coefs[3])
    bswir2 = img.select(bands[4]).multiply(coefs[4])
    albedo = bblue.add(bred).add(bnir).add(bswir).add(bswir2).add(coefs[5])
    albedo = albedo.rename('albedo')
    return albedo

def add_albedo_liang(img:ee.Image)->ee.Image:
    """Adds shortwave albedo (Liang, 2000) to a Landsat 7*, 8, or 9 image. 
    *Bands are renamed in place to match Landsat 8/9, but not returned. 
    bands: blue, red, nir, swir, swir2
    """
    band_names=['SR_B2', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    bands_l7=['SR_B1', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
    spacecraft_id = ee.String(img.get('SPACECRAFT_ID'))
    sel_bands = ee.Algorithms.If(spacecraft_id.equals('LANDSAT_7'),bands_l7, band_names)
    alb = albedo_liang(img.select(sel_bands, band_names)).rename("albedo")
    return img.addBands(alb.clamp(0,1))

def add_rad_temp(img:ee.Image)->ee.Image:
    """Adds the "radiometric_temperature" band to a Landsat 7*, 8, or 9 image. 
    *Bands are renamed in place to match Landsat 8/9, but not returned. 
    bands: surface temperature
    """
    spacecraft_id = ee.String(img.get('SPACECRAFT_ID'))
    thermal_band = ee.Algorithms.If(spacecraft_id.equals('LANDSAT_7'),"ST_B6", "ST_B10")
    return img.addBands(img.select([thermal_band], ["ST_B10"])
                        .rename("radiometric_temperature"))


def add_lai_loglinear(img:ee.Image)->ee.Image:
    """Adds a leaf area index (LAI*) band to a Lansdat image (requires NDVI band). 

    * "log-linear" method (e.g., Fisher et al., 2008):
        LAI = (-ln(1-fIPAR)/kPAR where fIPAR=1-0.05*NDVI) 

    Note: LAI is clamped to (0,7)
    """
    from geeet.vegetation import compute_lai
    lai = compute_lai(img.select("NDVI")).rename("LAI")
    return img.addBands(lai.clamp(0,7))


def add_lai_houborg2018(img:ee.Image)->ee.Image:
    """Adds a leaf area index (LAI**) band to a Landsat 7*, 8, or 9 image. 
    *Bands are renamed in place to match Landsat 8/9, but not returned. 

    ** "houborg2018" multi-vegetation index method (Houborg et al., 2018):

    Rule 1 (MSR <= 1.1384):
    LAI = -1.852 - 0.456 SR + 2.45 NDVI + 15.87 NDVI^2 + 0.8 EVI2 + 31.56 EVI2^2
        - 3.64 OSAVI - 36.35 OSAVI^2 + 1.5 EVI - 3.3 EVI^2 + 3.8 MSR
        - 4.38 NDWI - 5.96 NDWI^2 - 0.38 NDWI2 + 0.27 NDWI2^2
    
    Rule 2 (MSR > 1.1384):
    LAI = 3.061 - 0.004 SR + 87.32 NDVI - 19.65 NDVI^2 + 94.43 EVI2 - 29.5 EVI2^2
        - 172.66 OSAVI + 38.7 OSAVI^2 - 0.87 EVI - 2.95 EVI^2 + 3.45 MSR
        - 5.34 NDWI - 2.35 NDWI^2 - 18.08 NDWI2 + 14.54 NDWI2^2  

    Note: LAI is clamped to (0,7)
    """
    from geeet.vegetation import lai_houborg2018
    band_names=['SR_B2', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    bands_l7=['SR_B1', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
    spacecraft_id = ee.String(img.get('SPACECRAFT_ID'))
    sel_bands = ee.Algorithms.If(spacecraft_id.equals('LANDSAT_7'),bands_l7, band_names)
    img_renamed = img.select(sel_bands, band_names) 
    lai = (lai_houborg2018(
        blue = img_renamed.select("SR_B2"),
        red  = img_renamed.select("SR_B4"),
        nir  = img_renamed.select("SR_B5"),
        swir1= img_renamed.select("SR_B6"), 
        swir2= img_renamed.select("SR_B7"),
        ).rename("LAI")
    )
    return img.addBands(lai.clamp(0,7))

def collection(
    date_start: Union[datetime.datetime, ee.Date, int, str, Any],
    date_end: Union[datetime.datetime, ee.Date, int, str, Any],
    region: Union[Dict[str, Any], ee.Geometry],
    sat_list:List[str] = ["LANDSAT_7", "LANDSAT_8", "LANDSAT_9"],
    max_cc:Union[int, float] = None,        
    exclude_pr:bool = None,
    include_pr:bool = None,
    ndvi = True,
    albedo:Union[None, Literal["liang2001", "tasumi2008"]] = "liang2001", 
    rad_temp = True,
    cfmask = True, 
    era5:bool = False,
    lai:Union[None,Literal["log-linear", "houborg2018"]]=None,
    timeZone:str = "UTC"
)-> ee.ImageCollection:
    """Prepares a merged landsat collection

    Includes LE07, LC08, and LC09 collection 02 level 2 products.

    - Filters the collections to the specified date range and region. 
    - Scaling factors are applied to optical (SR_*) and thermal (ST_*) bands. 
    - Optionally includes additional albedo, NDVI, radiometric_temperature, cloud_cover*, and LAI bands. 

    Does not apply cloud mask; use `cfmask` to create a mappable function.

    Args: 
        date_start: The start date to retrieve the data (see ee.Date)
        date_end: The end date to retrieve the data (see ee.Date)
        region: The region of interest (see ee.ImageCollection.filterBounds)
        sat_list: A list, subset of ["LANDSAT_7", "LANDSAT_8", "LANDSAT_9"] but not empty.
        Indicates which satellites to include in the image collection.
        exclude_pr: A list of path/rows to exclude given as a list. E.g.: [[170,40]] will exclude PATH/ROW 170040
        include_pr: A list of path/rows to include given as a list. Any path/row not included here will be excluded.
        ndvi: Include NDVI as an additional band, using the NIR and Red bands. Defaults to True. 
        albedo: None, "liang2001", or "tasumi2008":

            * None: do not include albedo
            * "liang2001": Include albedo as an additional band based on Liang et al. 2001
            * "tasumi2008": Include albedo as an additional band based on Tasumi et al. 2008
            
        rad_temp: Include radiometric_temperature (e.g., required for TSEB) as an additional band. Defaults to True. 
        cfmask: Include cloud_cover as an additional band. Defaults to True. 
        era5: Include ERA5 meteorology. Defaults to False. Required for a landsat-ERA5 TSEB workflow.
        lai: None, "log-linear", or "houborg2018":
        
            * None: do not include LAI
            * "log-linear": Include LAI as an additional band using the NDVI-based relationship as in 
                PT-JPL (Fisher et al., 2008): LAI=(-ln(1-fIPAR)/kPAR where fIPAR=1-0.05*NDVI).
                NDVI will be included even if ndvi is set to False.  
            * "houborg2018": Include LAI as an additional band using the multi-spectral rule-based
                relationship from Houborg et al., 2018
        timeZone: see https://www.joda.org/joda-time/timezones.html. Used to set the
        "time" property in the image. 
    Returns: ee.ImageCollection
    """
    import ee
    from .join import landsat_ecmwf
    from .parsers import feature_collection

    region = feature_collection(region)

    filter_collection = ee.Filter.And(
        ee.Filter.bounds(region),
        ee.Filter.date(date_start, date_end)
    )
    L7_collection = (
        ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
        .filter(filter_collection)
        .map(set_index)
    )
    L8_collection = (
        ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        .filter(filter_collection)
        .map(set_index)
    )
    L9_collection = (
        ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
        .filter(filter_collection)
        .map(set_index)
    )

    collection = (L7_collection.merge(L8_collection)
    .merge(L9_collection)
    .filter(ee.Filter.inList("SPACECRAFT_ID", ee.List(sat_list))))
    
    if max_cc:
        collection = collection.filter(ee.Filter.lte("CLOUD_COVER", max_cc))

    if exclude_pr is not None:
        collection = collection.filter(ee.Filter.Or([
                ee.Filter.And(
                    ee.Filter.eq("WRS_PATH", pr[0]),
                    ee.Filter.eq("WRS_ROW", pr[1]))
            for pr in exclude_pr])
            .Not())

    if include_pr is not None:
         collection = collection.filter(ee.Filter.Or([
                ee.Filter.And(
                    ee.Filter.eq("WRS_PATH", pr[0]),
                    ee.Filter.eq("WRS_ROW", pr[1]))
            for pr in include_pr])
            )       

    collection = collection.map(scale_SR)

    if(ndvi):
        collection = collection.map(add_ndvi)

    if albedo is not None:
        if albedo=="liang2001":
            collection = collection.map(add_albedo_liang)
        if albedo=="tasumi2008":
            collection = collection.map(add_albedo_tasumi)

    if(rad_temp):
        collection = collection.map(add_rad_temp)

    if(cfmask):
        collection = collection.map(cloud_mask)

    if lai is not None:
        if lai=="log-linear":
            if not ndvi:
                collection = collection.map(add_ndvi)
            collection = collection.map(add_lai_loglinear)

        if lai=="houborg2018":
            collection = collection.map(add_lai_houborg2018)

    if era5:
        ## Meteorogical data from ECMWF collection -- hourly
        # https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY#bands)
        # see geeet.eepredefined.py for more details. 
        meteo_bands = geeet.eepredefined.MeteoBands.ECMWF_ERA5_HOURLY_TSEB
        meteo_prep = geeet.eepredefined.MeteoPrep.ECMWF_ERA5_HOURLY_TSEB
    
        meteo_collection = (ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
            .filterDate(date_start, date_end)
            .select(*meteo_bands)
            .map(meteo_prep))
    
        collection = landsat_ecmwf(
            collection, meteo_collection)

        # Set ERA5 measuring heights (zU, zT)
        def set_Z(img):
            return img.set(dict(zU=10, zT=2))

        collection = collection.map(set_Z)

    # Set additional image properties
    def set_additional_properties(img):
        d = ee.Date(img.date())
        doy = d.getRelative('day','year').add(1)  # from 0-based to 1-based. 
        time = d.getFraction('day', timeZone=timeZone).multiply(24)
        return img.set(dict(doy=doy, time=time, 
                            viewing_zenith=0) # Landsat has nadir view.
                            )
    
    collection = collection.map(set_additional_properties)

    return collection

def tseb_series(**kwargs)->Callable:
    """geeet.tseb tseb_series wrapper
    """
    from geeet.tseb import tseb_series as t
    def f(img):
        return t(img, **kwargs)
    return f


def extrapolate_LE(img):
    """
    Mappable function to extrapolate instantaneous LE to ET (mm/day)
    """
    R = geeet.solar.rad_ratio(img)
    LE = img.select("LE")
    ET = LE.multiply(R).rename("ET")
    return img.addBands(ET)

def mapped_collection(workflow:List[Callable], *args, **kwargs):
    """Map a custom algorithm defined by `workflow` onto a landsat collection.

    Args: 
    - workflow: A list of mappable functions
    - args: positional arguments for `landsat.collection`
    - kwargs: keyword arguments for `landsat.collection`
    """
    coll = collection(*args, **kwargs)
    for f in workflow:
        coll = coll.map(f)
    return coll


def mapped_reduced(workflow:List[Callable], 
                   feature_collection, landsat_kwargs, reducer_kwargs, 
                   **kwargs):
    """Map a custom algorithm defined by `workflow` onto a landsat collection 
    and reduce it (ee.ImageCollection-> ee.FeatureCollection)

    Args: 
    - workflow: A list of mappable functions
    - feature_collection: The feature collection use to reduce the image collection.
    - landsat_kwargs: Keyword arguments for `landsat.collection`
    - reducer_kwargs: Keyword arguments for `ee.Reducer.mean`
    - **kwargs: Keyword arguments for `reducers.image_collection`
    """
    from .reducers import image_collection
    return image_collection(feature_collection,
        mapped_collection(workflow, **landsat_kwargs),
        reducer_kwargs=reducer_kwargs,
        **kwargs
    )

def mapped_export(workflow:List[Callable], 
                   feature_collection, 
                   landsat_kwargs, 
                   reducer_kwargs, 
                   export_kwargs,
                   to:Literal["drive","cloudStorage"]= None,
                   **kwargs):
    """Map a custom algorithm defined by `workflow` onto a landsat collection,
    reduce it (ee.ImageCollection-> ee.FeatureCollection), and export it
    either to drive or cloudStorage

    Args: 
    - workflow: A list of mappable functions
    - feature_collection: The feature collection use to reduce the image collection.
    - landsat_kwargs: Keyword arguments for `landsat.collection`
    - reducer_kwargs: Keyword arguments for `ee.Reducer.mean`
    - export_kwargs: keyword arguments for ee.batch.Export.to(Drive | cloudStorage)
    - to: where to export: "drive" or "cloudStorage"
    - **kwargs: Keyword arguments for `reducers.image_collection`
    """
    output = mapped_reduced(workflow,
                    feature_collection,
                    landsat_kwargs,
                    reducer_kwargs,
                    **kwargs
                    )
    feature_properties=kwargs.get('feature_properties', [])
    img_properties=kwargs.get('img_properties', [])
    mean_bands = kwargs.get('mean_bands', [])
    sum_bands = kwargs.get('sum_bands', [])
    selectors = (["date"]+
               feature_properties +
               img_properties+
               mean_bands+
               sum_bands
               )
    if to=="drive":
        task = ee.batch.Export.table.toDrive(
            collection=output,
            selectors=selectors,
            **export_kwargs
        )
    if to=="cloudStorage":
        task = ee.batch.Export.table.toCloudStorage(
            collection=output,
            selectors=selectors,
            **export_kwargs
        )
    task.start()
    return task

def geesebal_compatibility(img:ee.Image)->ee.Image:
    """
    Mappable function to ensure compatibility with geeSEBAL

    - Optical bands are expected to have a 10000 scale factor, and named
    "B", "GR", "R", "NIR", "SWIR_1", "SWIR_2"
    - Albedo is expected as "ALFA"
    - The thermal band is expected to have a 10 scale factor, and named
    "BRT"
    - The image is expected to have the SOLAR_ZENITH_ANGLE and SATELLITE properties
    (as in Collection 01)
    - "T_RAD" is expected in W/(m^2*sr*um)/DN units
    """
    band_names = ["B", "GR", "R", "NIR", "SWIR_1", "SWIR_2"]
    oli_names = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
    etm_names = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
    spacecraft_id = ee.String(img.get('SPACECRAFT_ID'))
    sel_bands = ee.Algorithms.If(spacecraft_id.equals('LANDSAT_7'),etm_names, oli_names)
    normalized = img.select(sel_bands).rename(band_names).multiply(10000)  

    return (img
        .addBands(img.select("ST_TRAD").multiply(0.001).rename("T_RAD")) 
        .addBands(img.select("albedo").rename("ALFA"))   
        .addBands(img.select("radiometric_temperature").multiply(10).rename("BRT"))
        .addBands(normalized) 
        .set({
            "SOLAR_ZENITH_ANGLE": ee.Number(90).subtract(img.get("SUN_ELEVATION")),
            "SATELLITE": img.get("SPACECRAFT_ID"),
        })
    )

def albedo_liang_vis(img,
coefs = [0.443, 0.317, 0.240],
bands = ['SR_B2', 'SR_B3', 'SR_B4']):
    """
    Liang (2001) albedo (visible) parameterization
    see albedo_liang
    """
    bblue = img.select(bands[0]).multiply(coefs[0])
    bgreen = img.select(bands[1]).multiply(coefs[1])
    bred = img.select(bands[2]).multiply(coefs[2])
    albedo = bblue.add(bgreen).add(bred)
    return albedo.rename('albedo_vis')

def albedo_liang_nir(img,
coefs = [0.693, 0.212, 0.116, -0.003],
bands = ['SR_B5', 'SR_B6', 'SR_B7']):
    """
    Liang (2001) albedo (nir) parameterization
    see albedo_liang
    """
    bnir = img.select(bands[0]).multiply(coefs[0])
    bswir = img.select(bands[1]).multiply(coefs[1])
    bswir2 = img.select(bands[2]).multiply(coefs[2])
    albedo = bnir.add(bswir).add(bswir2).add(coefs[3])
    return albedo.rename('albedo_nir')
