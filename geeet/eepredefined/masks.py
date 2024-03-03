"""
Custom masks (based on NDVI, positive LE, fractional vegetation cover)
"""
import ee
from typing import Callable

def apply_static_mask(mask:str, bandNames:list)->Callable:
    def apply_mask(img:ee.Image)->ee.Image:
        """Returns img with the given mask (str) used to 
        update the mask on selected bands (bandNames)
        Requires bands: [mask]
        """
        imgBands = (img.select(bandNames)
        .updateMask(img.select(mask))
        )
        return img.addBands(imgBands, overwrite=True)

    return apply_mask


def Fndvi_mask(NDVI_BARE_GROUND=0.2)->Callable:
    def ndvi_mask(img:ee.Image)->ee.Image:
        """Returns a mask based on a threshold for NDVI
        Requires bands: "NDVI"
        Adds band: "vegetation_mask"
        """
        return img.addBands(
           img.select("NDVI").gt(NDVI_BARE_GROUND) 
           .rename("vegetation_mask")
        )
    return ndvi_mask


def Ffvc(fvc:ee.ImageCollection)->Callable:
    """Returns a mappable function to add fractional vegetation cover (fvc)
    
    Given an image collection of a single-band image (fvc), 
    this function adds fvc to an image for the given year, 
    **only** within the **unobserved** regions. 

    Requires band: "cloud_cover"
    
    Adds band: "fvc"
    """
    fvc = ee.ImageCollection(fvc)
    def f(img:ee.Image)->ee.Image:
        year=ee.Number(img.date().get("year"))
        return img.addBands(
            fvc.filter(ee.Filter.calendarRange(year, year, "year"))
            .first()
            .updateMask(
                img.select("cloud_cover")  
                .unmask(1)
            )
            .rename("fvc") 
        )
    return f


def positive_LE_mask(img:ee.Image)->ee.Image:
    """
    Mask to keep only positive LE values. 
    """
    return img.addBands(
        img.select("LE").gt(0)
        .rename("positive_le_mask")
    )