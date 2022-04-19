"""
Optional module to define some useful ee functions to join specific collections. 
"""
import ee

# Add a datetime property to the landsat data in the same format as ECMWF
def add_ECMWF_datetime(img):
    d = img.get('system:time_start')
    d_parsed = ee.Date(d).format("yyyyMMdd'T'HH")
    img = img.set({'system:datetime': d_parsed})
    return img

def landsat_ecmwf(Sat_collection, Meteo_collection):
    sat_data = Sat_collection.map(add_ECMWF_datetime)

    # Join the two collections using the datetime property. 
    filterByDateTime = ee.Filter.equals(leftField='system:datetime', rightField='system:index')
    joinByDateTime = ee.ImageCollection(ee.Join.inner().apply(sat_data, Meteo_collection, filterByDateTime))
    def get_img(feature):
        return ee.Image.cat(feature.get('primary'), feature.get('secondary'))

    et_inputs = joinByDateTime.map(get_img)
    return et_inputs