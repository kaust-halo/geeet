"""
Custom pixel area mappable functions. 
"""
import ee

def feature_area(img: ee.Image)->ee.Image:
    """Returns feature_area (pixelArea) as a band
    Reduce this band using ee.Reducer.sum() to get 
    the total area in m².

    Requires bands: None
    Adds bands: "feature_area"
    """
    return img.addBands(
        ee.Image.pixelArea().rename("feature_area")
    )


def unobserved_area(img: ee.Image)->ee.Image:
    """Returns unobserved_area (pixelArea) as a band
    
    For Landsat images, unobserved area is due to 
    cloud/cloud shadow mask and Landsat 7 slc error 
    gaps (stripes). 

    Reduce this band using ee.Reducer.sum() to get 
    the total unobserved area in m².
    
    Requires band: "cloud_cover", "feature_area"
    
    Adds band: "unobserved"
    """
    feature_pixel_area = img.select("feature_area")
    return img.addBands(
        feature_pixel_area.updateMask(
            img.select("cloud_cover")
            .unmask(1)
        )
        .rename("unobserved_area")
    )


def observed_veg_area(img:ee.Image)->ee.Image:
    """Returns observed_vegetation_area (pixelArea) as a band

    Requires band: "vegetation_mask", "feature_area"
    
    Adds band: "observed_vegetation_area"
    """
    feature_pixel_area = img.select("feature_area")
    return img.addBands(
        feature_pixel_area.updateMask(
            img.select("vegetation_mask")
        )
        .rename("observed_vegetation_area")
    )

def unobserved_veg_area(img:ee.Image)->ee.Image:
    """Adds the unobserved vegetation area band
    Requires band: "fvc", "feature_area"
    Adds band: "unobserved_vegetation_area"
    """
    feature_pixel_area = img.select("feature_area")
    return img.addBands(
        feature_pixel_area.updateMask(
            img.select("fvc")
        )
        .rename("unobserved_vegetation_area")
    )

