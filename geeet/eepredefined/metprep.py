"""
Optional module defining useful ee datasets, bands, and
functions for mapping to an Image Collection
"""
class MeteoBands:
    ECMWF_ERA5_HOURLY_TSEB=(
    [
    'surface_pressure', 
    'temperature_2m', 
    'u_component_of_wind_10m',
    'v_component_of_wind_10m',
    'surface_solar_radiation_downwards_hourly',
    'surface_thermal_radiation_downwards_hourly'
    ],
    [
    'surface_pressure',
    'air_temperature', 
    'u_component_of_wind_10m',
    'v_component_of_wind_10m',
    'surface_solar_radiation_downwards_hourly',
    'surface_thermal_radiation_downwards_hourly'
    ]
    )

class MeteoPrep:
    def ECMWF_ERA5_HOURLY_TSEB(img):
        # wind speed: square root of sum of squares
        u = img.select('u_component_of_wind_10m')
        v = img.select('v_component_of_wind_10m')
        wind = ((u.pow(2)).add(v.pow(2))).pow(0.5).rename('wind_speed')

        # solar and thermal radiation conversion to W/m2 and renaming bands
        Sdn = img.select('surface_solar_radiation_downwards_hourly').divide(3600.0).rename('solar_radiation')
        Ldn = img.select('surface_thermal_radiation_downwards_hourly').divide(3600.0).rename('thermal_radiation')
        return img.addBands(wind).addBands(Sdn).addBands(Ldn)