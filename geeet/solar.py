"""Solar radiation and energy balance related functions"""
from geeet.common import is_img, is_eenum

def std_meridian(longitude=None):
    '''
    Get the closest 15-degree meridian for a given Longitude
    For a numpy array output, longitude is a required input.
    For an ee.Image output, longitude is an optional input.   
    '''
    import numpy as np

    if longitude is None:
        from ee import Image
        lonlat_img = Image.pixelLonLat()
        longitude = lonlat_img.select('longitude')

    if is_img(longitude):
        stdMerid = longitude.add(187.5).divide(15).int().multiply(15).subtract(180)
    else:
        longitude = np.array(longitude)
        stdMerid = (((longitude + 187.5)/15).astype(int))*15-180

    return stdMerid

def eqn_time(doy):
    '''
    Function to calculate the equation of time, in hours
    Input: doy (np.array or ee.Number)
    Output: eot (np.array or ee.Number; matches the input type)
    '''
    import numpy as np
    DTOR = np.deg2rad(1) # constant to convert degrees to radians
    # Compute the value of the equation of time in hours eq
    if is_eenum(doy):
        f = doy.multiply(0.9856).add(279.575).multiply(DTOR)
        sinf = f.sin()
        sin2f = f.multiply(2).sin()
        sin3f = f.multiply(3).sin()
        sin4f = f.multiply(4).sin()
        cosf = f.cos()
        cos2f = f.multiply(2).cos()
        cos3f = f.multiply(3).cos()
        sin_term = sinf.multiply(-104.7).add(sin2f.multiply(596.2)).add(sin3f.multiply(4.3)).add(sin4f.multiply(-12.7))
        cos_term = cosf.multiply(-429.3).add(cos2f.multiply(-2.0)).add(cos3f.multiply(19.3))
        eot = sin_term.add(cos_term).divide(3600.0)
    else:
        f = (279.575 + 0.9856*doy)*DTOR
        eot = (-104.7*np.sin(f) + 596.2*np.sin(2*f) + 4.3*np.sin(3*f) - 12.7*np.sin(4*f) - 429.3*np.cos(f) - 2.0*np.cos(2*f) + 19.3*np.cos(3*f))/3600.0
    return eot

def compute_tnoon(doy, Lon = None, band_name = None):
    '''
    Function to compute the solar noon time in decimal hours.
    (Not accounting for daylight savings)
    Inputs:
        - doy (numpy array or ee.Number): the day of the year. 
        - Lon* (numpy array or ee.Image): the pixel longitudes.
       
    *For a numpy array output, longitude is a required input;
     for an ee.Image output, longitude is an optional input.   
    Outputs: 
        - T_noon (numpy array or ee.Image): the solar noon.
    References
    ----------
    Campbell and Norman, 1998
    '''

    import numpy as np
    from geeet.common import is_img

    if Lon is None:
        from ee import Image
        lonlat_img = Image.pixelLonLat()
        Lon = lonlat_img.select('longitude')

    Std_meridian = std_meridian(Lon)

    equation_time = eqn_time(doy)

    if is_img(Lon):
        from ee import Image
        LC = (Lon.subtract(Std_meridian)).multiply(4).divide(60)
        T_noon = Image(12.0).subtract(LC).subtract(equation_time)
        if band_name:
            T_noon = T_noon.rename(band_name)
    else:
        # Compute the time of solar noon eq (11.3)
        LC = (4*(Lon - Std_meridian))/60.0 # correction in hours
        T_noon = 12.0 - LC - equation_time
    
    return T_noon

def compute_solar_declination(doy):
    """
    Function to compute the solar declination angle for a given day of year
    Input: doy (np.array or ee.Number)
    Output: solar_declination (np.array or ee.Number; matches the input type) in radians
    """
    import numpy as np
    DTOR = np.deg2rad(1) # constant to convert degrees to radians
    if is_eenum(doy):
        from ee import Number
        doy_scaled = doy.multiply(0.9856)
        sin1 = doy_scaled.add(356.6).multiply(DTOR).sin()
        sin2 = sin1.multiply(1.9165).add(doy_scaled).add(278.97).multiply(DTOR).sin()
        sin3 = Number(np.sin(23.45*DTOR))
        solar_declination = sin2.multiply(sin3).asin()
    else:
        solar_declination = np.arcsin(np.sin(23.45*DTOR)*np.sin(DTOR*(278.97 + 0.9856*doy + 1.9165*np.sin((356.6 + 0.9856*doy)*DTOR))))
    return solar_declination

def compute_solar_angles(doy = None, time = None, longitude = None, latitude = None):
    '''
    Function to compute the sun Zenith and Azimuth angles in degrees 
    Inputs:
        - doy (ee.Number or numpy array): day of year
        - time (ee.Number or numpy array): time of measurement (decimal hours, local) 
        - longitude (ee.Number or numpy array): only required if doy is numpy array. 
        - latitude (ee.Number or numpy array): only required if doy is numpy array. 
    Outputs:
        - solar_angles (tuple or ee.Image): tuple containing numpy arrays with the following
                components, or ee.Image containing the following bands:
            - zenith: (degrees)
            - azimuth: (degrees) 
    '''
    import numpy as np 
    from geeet.common import is_eenum

    DTOR = np.pi/180.0 # constant to convert degrees to radians
    RTOD = 180.0/np.pi # constant to convert radians to degrees

    if is_eenum(doy):
        from ee import Image
        t_noon = compute_tnoon(doy)
    else:
        t_noon = compute_tnoon(doy, longitude)

    # Estimate solar declination in radians
    solar_declination = compute_solar_declination(doy) # in radians; ee.Number or np.array
      # tnoon = Local time when Apparent solar time (AST) is noon
    # tnoon = 12 - EOT - LC  where LC is longitude correction = (4/60)*(Lon - Std meridian)
    # AST = Local time + EOT + LC 
    # 12 = tnoon + EOT + LC
    # 12 - tnoon = EOT + LC
    # So we can get EOT + LC from tnoon
    # and then apply it to get AST
    # AST  = Local time + 12 - tnoon
    

    if is_eenum(doy):
        AST = Image(time.add(12)).subtract(t_noon)
        hour_angle = AST.subtract(12).multiply(15).multiply(DTOR)
        cos_hour_angle = hour_angle.cos()
        sin_hour_angle = hour_angle.sin()

        sin_solar = Image(solar_declination).sin() 
        cos_solar = Image(solar_declination).cos() 
        lonlat_img = Image.pixelLonLat()
        latitude = lonlat_img.select('latitude')

        lat_rad = latitude.multiply(DTOR)
        sin_lat = lat_rad.sin()
        cos_lat = lat_rad.cos()
        sin_term = sin_solar.multiply(sin_lat) 

        cos_term = cos_hour_angle.multiply(cos_lat).multiply(cos_solar)
        cos_zenith = sin_term.add(cos_term)
        zenith = cos_zenith.acos().multiply(RTOD)

        altitude_angle = Image(90).subtract(zenith).multiply(DTOR)
        cos_altitude = altitude_angle.cos()
        sin_azimuth = cos_solar.multiply(sin_hour_angle).divide(cos_altitude)
        azimuth = sin_azimuth.asin().multiply(RTOD)

        solar_angles = zenith.addBands(azimuth).rename(['zenith', 'azimuth'])

    else:
        # Apparent solar time : AST
        AST = time + 12 - t_noon 
        # Hour angle is (AST-12)*15  (if AST is 12 the hour angle is 0deg)
        hour_angle = (AST-12)*15
        # cos(Zenith) = sin(Lat)sin(declination)  + cos(Lat)cos(declination)cos(hour angle)
        cos_zenith = np.sin(latitude*DTOR)*np.sin(solar_declination) + \
            np.cos(latitude*DTOR)*np.cos(solar_declination)*np.cos(hour_angle*DTOR)
        zenith = np.arccos(cos_zenith)*RTOD
        # sin(azimuth) = cos(declination)*sin(hour angle) / cos(altitude angle)
        # where altitude angle is complementary to zenith (90 - zenith)
        altitude_angle = 90-zenith
        sin_azimuth = np.cos(solar_declination)*np.sin(hour_angle*DTOR)/np.cos(altitude_angle*DTOR)
        azimuth = np.arcsin(sin_azimuth)*RTOD
        solar_angles = [zenith, azimuth]

    return solar_angles

def compute_sunset_sunrise(img = None, doy = None, longitude = None, latitude = None):
    '''
    Function to compute the sunset and sunrise times 
    Inputs:
        - img (ee.Image) with the following bands or properties:
            - doy           (property) : day of year
            - longitude     (band)
            - latitude      (band)
        or the following numpy arrays as keywords:
            - doy: day of year
            - longitude
            - latitude

    Outputs:
        - solar_times (list or ee.Image): list containing numpy arrays with the following
                components, or ee.Image containing the following bands:
            - t_rise: time of sunrise (decimal time)
            - t_end:  time of sunset (decimal time)
    '''
    import numpy as np 

    DTOR = np.deg2rad(1) # constant to convert degrees to radians
    RTOD = np.rad2deg(1) # constant to convert radians to degrees

    if is_img(img):
        from ee import Number, Image
        doy = Number(img.get('doy'))
        t_noon = compute_tnoon(doy)
        Lat = img.select('latitude')
        # Estimate solar declination in radians
        solar_declination = compute_solar_declination(doy)
        lat_rad = Lat.multiply(DTOR)
        sin_lat = lat_rad.sin()
        cos_lat = lat_rad.cos()

        sin_solar = Image(solar_declination).sin() 
        cos_solar = Image(solar_declination).cos() 
        
        sin_term = sin_solar.multiply(sin_lat) 

        # Halfday length considering twilight
        # (using a zenitgh angle of the sun set to 96 degrees)
        cos_zs96 = Image(96*DTOR).cos()
        acos_zs96 = cos_zs96.subtract(sin_term).acos()
        halfday = acos_zs96.divide(cos_lat.multiply(cos_solar))
        halfday_h = halfday.multiply(RTOD).divide(15.0)  # in hours

        # Sunrise and sunset times
        t_rise = t_noon.subtract(halfday_h)
        t_end = t_noon.add(halfday_h)
        solar_times = t_rise.addBands(t_end)
        solar_times = solar_times.rename(['t_rise','t_end'])
    else:
        t_noon = compute_tnoon(doy, longitude)
        solar_declination = compute_solar_declination(doy)
        # Compute the halfday length considering twilight (set zs = 96 degrees) eq (11.6)
        halfday = np.arccos((np.cos(96*DTOR)-np.sin(latitude*DTOR)*np.sin(solar_declination))/(np.cos(latitude*DTOR)*np.cos(solar_declination)))
        halfday_h = halfday*RTOD/15.0 # converting to hours
        # Compute sunrise and sunset time eq (11.7)
        t_rise = t_noon - halfday_h
        t_end = t_noon + halfday_h
        solar_times = [t_rise, t_end]
    return solar_times

def rad_ratio(img=None, doy=None, time = None, longitude=None, latitude=None):
    '''
    Compute Jackson irradiance model (ratio of instantaneous radiation to daily radiation)
        Inputs:
        - img (ee.Image) with the following parameters:
            - doy (day of year)
            - time (time of observation)
        or
        the following numpy arrays as keyword arguments:
        doy (day of year)
        time (time of observation)
        longitude
        latitude 

        *The ee.Image should contain the property "system:time_start" 
        see https://developers.google.com/earth-engine/apidocs/ee-image-date
        and a time of observation property "time" in hours (local)
    Outputs:
       - Rs_ratio (numpy dstack or ee.Image): the computed ratio of instantaneous radiation to daily radiation
    '''
    import numpy as np 

    if is_img(img):
        from ee import Image, Number
        doy = Number(img.get('doy'))  
        lonlat_img = Image.pixelLonLat().set({'doy': doy})
        Time = Number(img.get('time'))  # time in hours (local time)
        solar_times = compute_sunset_sunrise(lonlat_img) 
        sunrise_time = solar_times.select('t_rise').multiply(3600)
        sunset_time = solar_times.select('t_end').multiply(3600)
        sun_seconds = sunset_time.subtract(sunrise_time)
        sun_obs = Image(Time.multiply(3600)).subtract(sunrise_time)
        denom = (sun_obs.divide(sun_seconds).multiply(np.pi).sin()).multiply(np.pi*1E6)
        Rs_ratio = sun_seconds.multiply(2).divide(denom)
        Rs_ratio = Rs_ratio.divide(2.45)  # Convert from MJ/m2day to mm/day 
    else:
        solar_times = compute_sunset_sunrise(doy=doy, longitude=longitude, latitude=latitude) 
        sunrise_time = solar_times[0]*3600
        sunset_time = solar_times[1]*3600
        N = sunset_time - sunrise_time
        t = time*3600 - sunrise_time
        Rs_ratio = 2*N/((10 ** 6) * np.pi * np.sin(np.pi * t/N))     
        Rs_ratio = Rs_ratio/2.45 # Convert to mm/day
    return Rs_ratio

def compute_g(doy, time, Rns, longitude = None, G_params = [0.31, 74000, 10800]):
    '''
    Function to compute the soil heat flux.
    Inputs:
        - doy (numpy array or ee.Number): the observation day (day of year)
        - time (numpy array or ee.Number): the observation local time in decimal hours.
        - Rns (numpy array or ee.Image): the net radiation to the soil.
    Optional_Inptus:
        - list or ee.List [float A, float B, float C] G_Params: the parameters for
          computing soil heat flux where A is the maximum ratio of G/Rns
          B reduces deviation of G/Rn to measured values, also thought of
          as the spread of the cosine wave and C is the phase shift between
          the peaks of Rns and G. B and C are in seconds.
        - longitude (numpy array): only required if Rns is a numpy array and not an ee.Image
    Outputs: 
        - G (numpy array or ee.Image): the soil heat flux.
    References 
    ----------
    Santanello and Friedl, 2003
    '''
    from geeet.common import is_img
    import numpy as np

    if is_img(Rns):
        from ee import Image, List, Number
        time = Number(time)
        doy = Number(doy)
        t_noon = compute_tnoon(doy)  # we ignore longitude, 
        # and this way we make sure t_noon is cast as an ee.Image 
        G_params = List(G_params)   # we make it a ee.List
        a = Number(G_params.get(0))
        b = Number(G_params.get(1))
        c = Number(G_params.get(2))
        t_g0 = Image(time).subtract(t_noon).multiply(3600.0)
        cos_term = t_g0.add(c).multiply(2.0*np.pi).divide(b)
        G = cos_term.cos().multiply(Rns).multiply(a).rename('soil_heat_flux')
    else:
        t_noon = compute_tnoon(doy, longitude) # longitude required for numpy array version.
        a = G_params[0]
        b = G_params[1]
        c = G_params[2]
        t_g0 = (time-t_noon)*3600.0
        cos_term = np.cos(2.0*np.pi*(t_g0+c)/b)
        G = a*cos_term*Rns
    return G


def compute_Rn(Sdn, Ldn, Albedo, Tr_K, fc, EmisVeg = 0.98, EmisGrd = 0.93):
    """
    Compute the net radiation using three components:
    (1) (1-albedo)*Sdn  where Sdn is the shortwave (solar) downwelling radiation
    (2) Ldn             where Ldn is the longwave (thermal) downwelling radiation
    (3) -eps*sigma*Tr^4 where eps is the emissivity of the surface, 
                        sigma is the Steffan-Boltzman constant, 
                        and Tr is the surface temperature.
    (1) and (3) as in Zhuang and Wu, 2015 (eqn 10), where
    the emissivity of the surface is calculated as:
    eps = fc Eveg + (1-fc)*Egrd  where Eveg and Egrd are the emissivity of vegetation 
    and ground (default values as given above) and fc is the fractional vegetation cover.

    
    Inputs:
        - Sdn (numpy array or ee.Image): Shortwave (solar) downwelling radiation (W/m2)
        - Ldn (numpy array or ee.Image): Longwave (thermal) downwelling radiation (W/m2)
        - Albedo (numpy array or ee.Image): Albedo of the surface
        - Tr_K (numpy array or ee.Image): Surface temperature, in Kelvin
        - fc (numpy array or ee.Image): Fractional vegetation cover
    Outputs: 
        - Rn (numpy array or ee.Image): Net radiation (W/m2).
    References
    ----------
    Zhuang, Q.; Wu, B. Estimating Evapotranspiration from an Improved Two-Source Energy 
    Balance Model Using ASTER Satellite Imagery. Water 2015, 7, 6673-6688. 
    https://doi.org/10.3390/w7126653 
    """
    sb = 5.670373e-8 # Steffan-Boltzman constant, in W⋅m−2⋅K−4
    import numpy as np
    if is_img(Sdn):
        from ee import Image
        Rn_short = (Image(1).subtract(Albedo)).multiply(Sdn)
        EmisSurf = (fc.multiply(EmisVeg)).add((Image(1).subtract(fc)).multiply(EmisGrd))
        Rn_long = Ldn.subtract(EmisSurf.multiply(sb).multiply(Tr_K.pow(4)))
        Rn = Rn_short.add(Rn_long)
    else:
        EmisSurf = (fc*EmisVeg + (1-fc)*EmisGrd)
        Rn = (1 - Albedo)*Sdn + Ldn - EmisSurf* sb * Tr_K ** 4
        Rn = np.array(Rn)
    return Rn
