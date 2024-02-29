"""
Custom parsers for use with earthengine-api.
"""
import ee

def feature_collection(input) -> ee.FeatureCollection:
    """Returns input as ee.FeatureCollection

    Input options:
        - ee.FeatureCollection
        - dict (json representation)
        - geopandas GeoDataFrame
        - string (ee asset)
        - string (local .shp or .geojson file: it must have an extension)

    > dict is cast to ee.FeatureCollection only if its "type" is 
    FeatureCollection, otherwise it's ignored (returned as is).
    """
    import os.path
    try:
        import geopandas as gpd
        if isinstance(input, gpd.GeoDataFrame):
            return feature_collection(eval(input.to_json()))
    except:
        pass

    if isinstance(input, ee.FeatureCollection):
        return input

    if isinstance(input, dict):
        if input["type"]=="FeatureCollection":
            return ee.FeatureCollection(input)
        else:
            return input
    
    if isinstance(input, str):
        _,ext = os.path.splitext(input)
        if ext: return feature_collection(gpd.read_file(input))
        return ee.FeatureCollection(input) 