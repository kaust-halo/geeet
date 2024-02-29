"""Reducers (ee.ImageCollection -> ee.FeatureCollection)
"""
import ee
from typing import List, Optional, Callable

def image(
    feature_collection: ee.FeatureCollection, 
    FILL_DICT: dict,
    reducer_kws: dict, 
    reducer: ee.Reducer,
    properties: List[str] =[],
    date_format:str ="YYYY-MM-dd'T'HH:mm:ss") -> Callable:
    """Returns a function to reduce (ee.Image.reduceRegions) an ee.Image
        - retrieves properties from the image
        - sets a 'date' property 
        - sets a fill value (if region is fully masked).
    """
    def f_reducer(img):
        def set_fill_values(feature):
            return feature.set(
                feature.toDictionary()
                .combine(FILL_DICT, False)
            )
        props = {} 
        for property in properties:
            props[property] = img.get(property)

        return (img.reduceRegions(
            collection = feature_collection,
            reducer = reducer, 
            **reducer_kws
            )
            .map(set_fill_values) 
            .map(lambda feature: feature.set({
                "date": ee.Date(img.date()).format(date_format)
                }))
            .map(lambda feature: feature.set(props))
        )
    return f_reducer

def image_collection(feature_collection:ee.FeatureCollection, 
                    img_collection: ee.ImageCollection,
                    mean_bands: List[str],
                    sum_bands:Optional[List[str]]=[],
                    img_properties:Optional[List[str]]=[],
                    feature_properties: Optional[List[str]]=[],
    reducer_kwargs:dict=dict(crs="EPSG:3857", scale=30),
    na_value=-1, 
    ):
    """Reduces an image collection into a feature collection

    Args:
        feature_collection: regions where the images will be reduced
        img_collection: the ee.ImageCollection to be reduced
        mean_bands: Bands that will be reduced by using ee.Reducer.mean()
        sum_bands: Optional bands to reduce using ee.Reducer.sum()
        img_properties: Optional properties from each image to keep.
        feature_properties: Optional properties from the feature collection to keep.
        reducer_kwargs: keyword arguments to be passed to the ee.Reducer (s).
            Defaults to 30m scale in Mercator projection.
        na_value: value to use for fully masked features
    """
    from .parsers import feature_collection as parsefc
    feature_collection = parsefc(feature_collection)

    usecols = (["date"]+
               feature_properties +
               img_properties+
               mean_bands+
               sum_bands
               )
    fill_bands = [x+"_mean" for x in mean_bands] + [x+"_sum" for x in sum_bands]
    FILL_DICT = {band: na_value for band in fill_bands}
    combined_reducer = ee.Reducer.mean().combine(ee.Reducer.sum(), "", True)
    reducer = image(feature_collection, FILL_DICT, reducer_kwargs, combined_reducer,
                    img_properties)
    return ee.FeatureCollection(img_collection
        .map(reducer)
        .flatten()
        .sort("date")
        .select(
            ["date"] + feature_properties + img_properties + fill_bands,
            usecols
        )
        )
