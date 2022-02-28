"""
Optional module to define some useful ee functions related to Landsat images processing. 
"""

def albedo_liang(img, 
coefs=[0.356, 0.130, 0.373, 0.085, 0.072, -0.0018],
bands = ['SR_B2', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']):
    """
    Liang (2001) albedo parameterization 
    for Landsat-8 data: 
    0.356*blue + 0.130*red + 0.373*nir + 0.085*swir + 0.072*swir2 - 0.0018

    Optional inputs (defaults to Landsat-8 coefficients and band names*):
    coefs: 6 coefficients for a Landsat sensor
    bands: names of the bands in the Landsat ee.Image
        (corresponding to the blue, red, nir, swir, and swir2 bands)

    *for collection 02, level 2 (SR) product. 

    TODO 
    add visible and nir parameterizations as well
    (needed for CABLE).
    visible: 
    0.443*blue + 0.317*green + 0.240*red
    nir:
    0.693*nir + 0.212*swir  + 0.116*swir2 - 0.003

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

def l8c02_cloud_mask(img):
    """
    Quality bit-based cloud/cloud-shadow mask for
    a Landsat Collection 02 ee.Image (either TOA or SR)
    Input: img (ee.Image)
    """
    # mask if QA_PIXEL has any of the 0,1,2,3,4 bits on
    # which correspond to: Fill, Dilated Cloud, Cirurs, Cloud, Cloud shadow
    # i.e. we want to keep pixels where the bitwiseAnd with 11111 is 0:
    qa_mask = img.select('QA_PIXEL').bitwiseAnd(int('11111',2)).eq(0)
    # Mask any over-saturated pixel as well: 
    saturation_mask = img.select('QA_RADSAT').eq(0)
    return img.updateMask(qa_mask).updateMask(saturation_mask)

def l8c02_mask_scale_SR(img):
    """
    Quality bit-based cloud/cloud-shadow mask for 
    a Landsat Surface reflectance image (Collection 02)
    based on the GEE code editor example 
    Examples/Cloud Masking/Landsat8 Surface Reflectance

    It also returns the scaled optical and thermal bands 
    (although the thermal bands will not be used)

    Input: img (ee.Image)
    """
    # Mask the image:
    img = l8c02_cloud_mask(img)
    # Scaling factors for Collection 02, level 2:
    opticalBands = img.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = img.select('ST_B.*').multiply(0.00341802).add(149.0)

    img = img.addBands(opticalBands, overwrite=True)\
        .addBands(thermalBands, overwrite=True)
    return img

def l8c02_add_inputs(img):
    """
    Adds the "albedo", "NDVI" and "radiometric_temperature" bands
    Albedo: shortwave albedo using the empirical coefficients of Liang (2001)
    LST: ST_B10 for "radiometric_temperature" 
    """
    albedo = albedo_liang(img)
    ndvi = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    lst = img.select('ST_B10').rename('radiometric_temperature')
    return img.addBands(albedo).addBands(ndvi).addBands(lst)