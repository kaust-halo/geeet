"""Functions common to et models and other utilities."""

import sys


#### Check if object is an ee object

def is_img(obj):
    '''
    Function to check if an object is an instance of ee.Image
    '''
    if 'ee' in sys.modules:
        return isinstance(obj, sys.modules['ee'].Image)
    else:
        return False

def is_eenum(obj):
    '''
    Function to check if an object is an instance of ee.Number
    '''
    if 'ee' in sys.modules:
        return isinstance(obj, sys.modules['ee'].Number)
    else:
        return False

#### Humidity calculations 

def teten(T):
    '''
    Compute Teten's formula for saturation water vapour pressure  (esat (T)) 
    Reference: 
    https://www.ecmwf.int/sites/default/files/elibrary/2016/17117-part-iv-physical-processes.pdf#subsection.7.2.1    

    Input: T (numpy array or ee.Image) Temperature at which to calculate esat
    '''
    import numpy as np
    a1 = 611.21 # in Pa
    a3 = 17.502 
    a4 = 32.19 # in K
    T0 = 273.16 # in K
    
    if is_img(T):
        T1 = T.subtract(T0) # in K
        T2 = T.subtract(a4) # in K
        esat = T1.divide(T2).multiply(a3).exp().multiply(a1)
    else:
        esat = a1*np.exp(a3*(T-T0)/(T-a4))
    return esat

def specific_humidity(img):
    '''
    Input: img (list of numpy arrays or ee.Image) with following bands/arrays:
        - dewpoint_temperature     (K)
        - surface_pressure         (Pa)

    Note: if the names of the image are not the same as defined above, 
    you can use img.select([old_names], [new_names]) to match the names above
    before calling this function.
    '''
    Rdry = 287.0597
    Rvap = 461.5250
    epsilon = Rdry/Rvap
    if is_img(img):
        T = img.select('dewpoint_temperature')
        esat = teten(T)
        P = img.select('surface_pressure')
        denom = P.subtract(esat.multiply(1-epsilon))
        Q = esat.multiply(epsilon).divide(denom)
    else:
        T = img[0]
        esat = teten(T)
        P = img[1]
        Q = epsilon*esat/(P-(1-Rdry/Rvap)*esat)
    return Q

def relative_humidity(img):
    '''
    Input: img (list of numpy arrays or ee.Image) with following bands/arrays:
        - temperature   (K)
        - dewpoint_temperature (K)
        - surface_pressure (Pa)
    Note: if the names of the image are not the same as defined above, 
    you can use img.select([old_names], [new_names]) to match the names above
    before calling this function.

    Output: 
        - Relative humidity (%) either as a numpy array
        or returned with the same image as an added band named 'relative_humidity'
    '''
    Rdry = 287.0597
    Rvap = 461.5250
    epsilon = Rdry/Rvap
    if is_img(img):
        T = img.select('temperature')
        P = img.select('surface_pressure')
        Q = specific_humidity(img)
        esat = teten(T)
        denom = Q.multiply(1/epsilon -1).add(1).multiply(esat)
        RH_band = P.multiply(Q).multiply(100/epsilon).divide(denom).rename('relative_humidity')
        RH = img.addBands(RH_band)
    else:
        T = img[0]
        D = img[1]
        P = img[2]
        esat = teten(T)
        Q = specific_humidity([D,P])
        RH = (P*Q*100/epsilon)/(esat*(1+Q*((1/epsilon) - 1)))
    return RH
    
#### Solar-related and heat flux computations

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
    Output: eot (np.array or ee.Number; whichever the input type is)
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
    Inputs:
        - doy (numpy array or ee.Number*): the day of the year. 
        - Lon** (numpy array or ee.Image): the pixel longitudes.
    * if running the model with a single ee.Image, doy could be ee.Number or a scalar value.
    However, for mapping an ET model as a function to a ee.ImageCollection, doy should be an ee.Number. 
       
    **For a numpy array output, longitude is a required input;
     for an ee.Image output, longitude is an optional input.   

    Outputs: 
        - T_noon (numpy array or ee.Image): the solar noon.
    References
    ----------
    Campbell and Norman, 1998
    '''

    import numpy as np
    from geeet.common import is_img, std_meridian

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
        LC = (4*(Lon - Std_meridian))/60.0 # compute latitude correction in hours
        T_noon = 12.0 - LC - equation_time
    
    return T_noon


def compute_sunset_sunrise(doy, Lon = None, Lat = None, t_noon = None, band_names = ['t_rise', 't_end']):
    '''
    Function to compute the sunset and sunrise times 
    Inputs:
        - doy (numpy array or ee.Number*): the day of the year. 
        - Lon** (numpy array or ee.Image): the pixel longitudes.
        - Lat** (numpy array or ee.Image): the pixel latitudes.
    * if running the model with a single ee.Image, doy could be ee.Number or a scalar value.
    However, for mapping an ET model as a function to a ee.ImageCollection, doy should be an ee.Number. 

    Optional inputs:
    **For a numpy array output, longitude and latitude are required outputs;
     for an ee.Image output, they are optional.   

        - t_noon (numpy array or ee.Image): solar noon (decimal time).
                                    If not provided, it will be computed using compute_tnoon
    Outputs:
        - solar_times (tuple or ee.Image): tuple containing numpy arrays with the following
                components, or ee.Image containing the following bands:
            - t_rise: time of sunrise (decimal time)
            - t_end:  time of sunset (decimal time)
            n.b. we could add the zenith angle of the sun here if needed. 
    '''
    import numpy as np 

    if Lon is None:
        from ee import Image
        lonlat_img = Image.pixelLonLat()
        Lon = lonlat_img.select('longitude')
        Lat = lonlat_img.select('latitude')

    if t_noon is None:
        from geeet.common import compute_tnoon
        t_noon = compute_tnoon(doy, Lon)

    DTOR = np.deg2rad(1) # constant to convert degrees to radians
    RTOD = np.rad2deg(1) # constant to convert radians to degrees

    # Estimate solar declination in radians
    if is_eenum(doy):
        import ee
        doy_scaled = doy.multiply(0.9856)
        sin1 = doy_scaled.add(356.6).multiply(DTOR).sin()
        sin2 = sin1.multiply(1.9165).add(doy_scaled).add(278.97).multiply(DTOR).sin()
        sin3 = ee.Number(np.sin(23.45*DTOR))
        solar_declination = sin2.multiply(sin3).asin()
    else:
        solar_declination = np.arcsin(np.sin(23.45*DTOR)*np.sin(DTOR*(278.97 + 0.9856*doy + 1.9165*np.sin((356.6 + 0.9856*doy)*DTOR))))
   
    if is_img(Lon):
        from ee import Image
        lat_rad = Lat.multiply(DTOR)
        sin_lat = lat_rad.sin()
        cos_lat = lat_rad.cos()

        sin_solar = Image(solar_declination).sin() 
        cos_solar = Image(solar_declination).cos() 
        
        sin_term = sin_solar.multiply(sin_lat) 

        # Zenith angle of the sun (to enable these lines of code if a model requires it)
        #cos_time = ((Image(Time_t).subtract(t_noon)).multiply(15*DTOR)).cos()
        #cos_term = cos_lat.multiply(cos_solar).multiply(cos_time)
        #cos_zs = sin_term.add(cos_term)
        #zs = cos_zs.acos()    # in radians

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
        solar_times = solar_times.rename(band_names)
        return solar_times
    else:
        # Compute the zenith angle of the sun eq (11.1) time_t must be in standard time (local time disregarding daylight savings adjustment)
        #cos_zs = np.sin(Lat*DTOR)*np.sin(solar_declination) +np.cos(Lat*DTOR)*np.cos(solar_declination)*np.cos((15*(Time_t - t_noon))*DTOR)

        # Solar angle is in radians since albedo_separation expects zs in radians.
        #zs = np.arccos(cos_zs)

        # Compute the halfday length considering twilight (set zs = 96 degrees) eq (11.6)
        halfday = np.arccos((np.cos(96*DTOR)-np.sin(Lat*DTOR)*np.sin(solar_declination))/(np.cos(Lat*DTOR)*np.cos(solar_declination)))
        halfday_h = halfday*RTOD/15.0 # converting to hours
        # Compute sunrise and sunset time eq (11.7)
        t_rise = t_noon - halfday_h
        t_end = t_noon + halfday_h
        solar = [t_rise, t_end]
        return solar

def rad_ratio(doy, Time, longitude=None, latitude=None):
    '''
    Compute Jackson irradiance model (ratio of instantaneous radiation to daily radiation)
    For a numpy array output, longitude and latitude are required inputs.
    For an ee.Image output, longitude and latitude are optional inputs. 

        Inputs:
        - doy (numpy array or ee.Number): the day of the year.
        - Time_T (numpy array): the time of the observation, in hours (e.g. 11 for 11am)
        - Lon (numpy array or ee.Image): the pixel longitudes. 
        - Lat (numpy array or ee.Image): the pixel latitudes.
    Outputs:
       - Rs_ratio (numpy dstack or ee.Image): the computed ratio of instantaneous radiation to daily radiation
    '''
    import numpy as np 

    # if longitude is not provided, we assume the user wants an ee.Image
    # in which case we create it using ee.Image.pixelLonLat()
    if (longitude is None) or (latitude is None):
        from ee import Image
        lonlat_img = Image.pixelLonLat()
        longitude = lonlat_img.select('longitude')
        latitude = lonlat_img.select('latitude')

    solar_times = compute_sunset_sunrise(doy, longitude, latitude)

    if is_img(longitude):
        from ee import Image
        sunrise_time = solar_times.select('t_rise').multiply(3600)
        sunset_time = solar_times.select('t_end').multiply(3600)
        sun_seconds = sunset_time.subtract(sunrise_time)
        sun_obs = Image(Time*3600).subtract(sunrise_time)
        denom = (sun_obs.divide(sun_seconds).multiply(np.pi).sin()).multiply(np.pi*1E6)
        Rs_ratio = sun_seconds.multiply(2).divide(denom)
        Rs_ratio = Rs_ratio.divide(2.45)  # Convert from MJ/m2day to mm/day 
    else:
        # Compute the simulated ratio of instantaneous incomming radiation to daily radiation
        sunrise_time = solar_times[0]*3600
        sunset_time = solar_times[1]*3600
        N = sunset_time - sunrise_time
        t = Time*3600 - sunrise_time
        Rs_ratio = 2*N/((10 ** 6) * np.pi * np.sin(np.pi * t/N)) # (MJ/m2day) / (W/m2)    
        Rs_ratio = Rs_ratio/2.45 # Convert from MJ/m2day to mm/day
    return Rs_ratio

def compute_g(Time, T_noon, Rns, band_name=None, G_Params = [0.31, 74000, 10800]):
    '''
    Function to compute the soil heat flux.
    Inputs:
        - Time (numpy array): the current time in decimal hours.
        - T_noon (numpy array or ee.Image): the solar noon time in decimal hours.
        - Rns (numpy array or ee.Image): the net radiation to the soil.
    Optional_Inptus:
        - list [float A, float B, float C] G_Params: the parameters for
          computing solid heat flux where A is the maximum ratio of G/Rns
          B reduces deviation of G/Rn to measured values, also thought of
          as the spread of the cosine wave and C is the phase shift between
          the peaks of Rns and G. B and C are in seconds.
    Outputs: 
        - G (numpy array or ee.Image): the soil heat flux.
    References 
    ----------
    Santanello et al., 2003

    Note: for this function, we are being more lenient for T_noon and Time
    If Rns is NOT an image, we assume T_noon and Time aren't either
    But if Rns is an image and T_noon or Time aren't, we will cast them to ee.Image
    '''

    import numpy as np

    a = G_Params[0]
    b = G_Params[1]
    c = G_Params[2]

    if is_img(T_noon):
        if is_img(Time):
            t_g0 = Time.subtract(T_noon).multiply(3600.0)
        else:
            import ee
            t_g0 = ee.Image(Time).subtract(T_noon).multiply(3600.0)
    else:
        t_g0 = (Time-T_noon)*3600.0

    if is_img(t_g0):
        cos_term = t_g0.add(c).multiply(2.0*np.pi/b)
        G = cos_term.cos().multiply(Rns).multiply(a)
    else:
        cos_term = np.cos(2.0*np.pi*(t_g0+c)/b)
        if is_img(Rns):
            G = Rns.multiply(cos_term).multiply(a)
        else:
            G = a*cos_term*Rns

    if is_img(G) and band_name:
        G = G.rename(band_name)

    return G


#### Functions to update the package from Github, taken and modified from geemap.common (credit to Qiusheng Wu)
import os, shutil
def update_package():
    """Updates the geeet package from the geeet GitHub repository without the need to use pip or conda.
    In this way, I don't have to keep updating pypi and conda-forge with every minor update of the package.

    """
    try:
        download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        clone_repo(out_dir=download_dir)

        pkg_dir = os.path.join(download_dir, "geeet-main")
        work_dir = os.getcwd()
        os.chdir(pkg_dir)

        if shutil.which("pip") is None:
            cmd = "pip3 install ."
        else:
            cmd = "pip install ."

        os.system(cmd)
        os.chdir(work_dir)

        print(
            "\nPlease comment out 'geeet.update_package()' and restart the kernel to take effect:\nJupyter menu -> Kernel -> Restart & Clear Output"
        )

    except Exception as e:
        raise Exception(e)

def clone_repo(out_dir=".", unzip=True):
    """Clones the geeet GitHub repository.

    Args:
        out_dir (str, optional): Output folder for the repo. Defaults to '.'.
        unzip (bool, optional): Whether to unzip the repository. Defaults to True.
    """
    from geemap.common import download_from_url
    url = "https://github.com/kaust-halo/geeet/archive/main.zip"
    filename = "geeet-main.zip"
    download_from_url(url, out_file_name=filename, out_dir=out_dir, unzip=unzip)