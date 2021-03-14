"""
This module contains functions to run the PT-JPL crop water use model.

The model functions are hybrid: they work both with numpy arrays or ee.Images*:
If the instance of the input is recognized as an ee.Image, the output is
computed as an ee.Image as well.
Otherwise, it is computed as a numpy array.

To reduce the dependencines of this package, 
the ee capabilities are optional. That means that 
in order to use the ee capabilities, the user must install the ee package:

conda install -c conda-forge earthengine-api
or 
pip install earthengine-api

*Exceptions:
    add_fapar - function intended only for ee.Images

References can be found at the end of this module
They can be printed in python using the following two functions:
geeet.ptjpl.cite() - main reference for this module
geeet.ptjpl.cite_all() - all references used for this module
"""

import sys
import numpy as np
try: 
    import ee
except Exception:
    pass

def is_img(obj):
    '''
    Function to check if an object is an instance of ee.Image
    '''
    if 'ee' in sys.modules:
        return isinstance(obj, sys.modules['ee'].Image)
    else:
        return False

def compute_fwet(RH, band_name=None):
    '''
    Function to compute the relative surface wetness.
    Inputs:
        - RH (numpy array or ee.Image): the relative humidity [0-100].
    Outputs: 
        - Fwet (numpy array or ee.Image): the relative surface wetness.
    References
    ----------
    Fisher et al., 2008
    '''
    if is_img(RH):
        Fwet = (RH.divide(100.0)).pow(4)
        if band_name:
            Fwet = Fwet.rename(band_name)
    else:
        Fwet = (RH/100.0) ** 4
    return Fwet


def compute_fapar(NDVI, NDVIsoil = 0.17, NDVIveg = 0.97, band_name=None):
    '''
    Function to compute the fraction of the photosynthetic 
    active radiation (PAR) absorbed by green vegetation cover.
    Inputs:
        - NDVI (numpy array or ee.Image): normalized vegetation index values.
    Optional Inputs:
        - NDVIsoil (numpy array or ee.Image): minimum value of NDVI over bare soil areas in the period.
        - NDVIveg (numpy array or ee.Image): maximum NDVI value for the vegetated areas in the period.
    Outputs: 
        - F_apar (numpy array or ee.Image): fraction of the PAR absorbed by
        green vegetation cover.
    References
    ----------        
    Carlson et al., 1995        

    Choudhury et al., 1994        

    Aragon et al., 2018      
    '''
    import numpy as np

    if is_img(NDVI):
        # if NDVIsoil or NDVIveg are float, conver to ee.Image:
        if not is_img(NDVIsoil):
            NDVIsoil = ee.Image(NDVIsoil)
        
        if not is_img(NDVIveg):
            NDVIveg = ee.Image(NDVIveg)
 
        F_apar = (NDVI.subtract(NDVIsoil)).divide(NDVIveg.subtract(NDVIsoil))
        if band_name:
            F_apar = F_apar.rename(band_name)
    else:
        F_apar=(NDVI-NDVIsoil)/(NDVIveg-NDVIsoil)
    return F_apar


def add_fapar(NDVI_image, NDVIsoil = 0.17, NDVIveg = 0.97):
    '''
    Function to add the computed fapar as a band to an ee.Image. It is useful to map
    this function to an ee.ImageCollection in order to reduce it to get 
    fapar_max
    Therefore, this function is intended **only for ee.ImageCollection**

    Inputs:
        - NDVI_image (ee.Image): NDVI image 
    Outputs: 
        - out_img (ee.Image): The input NDVI_image with the band 'fapar' added to it.
    '''
    out_img = NDVI_image.addBands(compute_fapar(NDVI_image, NDVIsoil, NDVIveg, band_name='fapar'))
    return out_img


def compute_fipar(NDVI, band_name=None):
    '''
    Function to compute the fraction of the photosynthetic 
    active radiation (PAR) intercepted by total vegetation cover.
    Inputs:
        - NDVI (numpy array or ee.Image): normalized vegetation index values.
    Outputs: 
        - F_ipar(numpy array or ee.Image): fraction of the PAR intercepted by
        total vegetation cover.
    References
    ----------
    Fisher et al., 2008
    '''

    import numpy as np

    m2 = 1.0
    b2 = -0.05

    if is_img(NDVI):
        m2 = ee.Image(m2)
        b2 = ee.Image(b2)
        F_ipar = (m2.multiply(NDVI)).add(b2)
        F_ipar = F_ipar.where(F_ipar.lt(0), 0)
        if band_name:
            F_ipar = F_ipar.rename(band_name)
    else:
        F_ipar = m2*NDVI + b2
        F_ipar = np.array(F_ipar)
        F_ipar[F_ipar < 0] = 0

    return F_ipar


def compute_fg(NDVI, band_name=None):
    '''
    Function to compute the green canopy fraction
    Inputs:
        - NDVI (numpy array or ee.Image):  normalized vegetation index values.
    Outputs: 
        - Fg (numpy array or ee.Image): the green canopy fraction.
    References
    ----------
    Fisher et al., 2008
    '''

    import numpy as np

    f_apar = compute_fapar(NDVI)
    f_ipar = compute_fipar(NDVI)

    if is_img(NDVI):
        Fg = f_apar.divide(f_ipar)
        Fg = Fg.where(Fg.gt(1), 1)
        Fg = Fg.where(Fg.lt(0), 0)
        if band_name:
            Fg = Fg.rename(band_name)
    else:
        Fg = f_apar/f_ipar
        Fg = np.array(Fg)
        Fg[Fg > 1] = 1
        Fg[Fg < 0] = 0
        Fg[f_ipar <= 0] = 0
    
    return Fg


def compute_ft_arid(T_a, band_name=None):
    '''
    Function to compute the plant temperature constraint Ft.
    This function is specific for arid environments! (Aragon et al. 2018)
    Inputs:
        - T_a (numpy array or ee.Image): the air temperature [C].
    Outputs: 
        - Ft (numpy array or ee.Image): the plant temperature constraint. 
    References
    ----------
    Potter et al., 1993        

    Aragon et al., 2018        
    '''

    import numpy as np

    if is_img(T_a):
        cst1 = ee.Image(1.0)
        exp_arg = (ee.Image(12.0).subtract(T_a)).multiply(0.2)
        Ft = cst1.divide(cst1.add(exp_arg.exp()))
        if band_name:
            Ft = Ft.rename(band_name)
    else:
        Ft = 1.0 /(1.0 + np.exp(0.2*(12 - T_a)))
        # Bound Ft
        Ft = np.array(Ft)
        Ft[Ft > 1] = 1
        Ft[Ft < 0] = 0

    return Ft


def compute_fm(F_apar, F_aparmax):
    '''
    Function to compute the plant moisture constraint Fm.
    Inputs:
        - F_apar (numpy array or ee.Image): photosynthetic active radiation (PAR) absorbed by green vegetation cover.
        - F_aparmax (numpy array or ee.Image): maximum fapar value (e.g. in one year)
        Either both inputs are numpy array, or ee.Image, but should not be mixed!
    Outputs: 
        - Fm (numpy array or ee.Image): the plant moisture constraint.
    References
    ----------
    Fisher et al., 2008
    '''

    import numpy as np

    is_fapar_img = is_img(F_apar) 
    is_fapar_max_img = is_img(F_aparmax) 
    
    if is_fapar_img != is_fapar_max_img:
        print('Either both inputs should be numpy array or ee.Img, but not mixed.')
        return

    if is_fapar_img:
        Fm = F_apar.divide(F_aparmax)
        Fm = Fm.where(Fm.lt(0),0)
        Fm = Fm.where(Fm.gt(1), 1)
    else:
        Fm = F_apar/F_aparmax
        # Bound Fm
        Fm = np.array(Fm)
        Fm[Fm < 0] = 0
        Fm[Fm > 1] = 1

    return Fm


def compute_gamma(Pressure, band_name = None):
    '''
    Function to compute the slope of the psychrometric constant.
    Inputs:
        - Pressure (numpy array or ee.Image): the atmospheric pressure [kPa]
    Outputs: 
        - Gamma (numpy array or ee.Image): the psychrometric constant [kPa C-1] 
    References
    ----------
    Allen et al., 1998
    '''

    Cp = 1.013*(10 ** -3) # specific heat at constant pressure
    e = 0.622 # ratio of molecular weight of water vapour to dry air
    L = 2.45 # latent heat of vaporization

    if is_img(Pressure):
        Gamma = (Pressure.multiply(Cp)).divide(e*L)
        if band_name:
            Gamma = Gamma.rename(band_name)
    else:
        Gamma = Cp*Pressure/(e*L)

    return Gamma


def compute_delta(Temp_C, band_name = None):
    '''
    Function to compute the slope of the relationship between 
    saturation vapour pressure and air temperature.
    Inputs:
        - Temp_C (numpy array or ee.Image): Temperature in Celsius.
    Outputs: 
        - Delta (numpy array or ee.Image): the slope of saturation vapour 
        pressure curve [KPa C-1].
    References
    ----------
    Allen et al., 1998
    See: http://www.fao.org/3/x0490e/x0490e07.htm#air%20temperature
    '''

    import numpy as np

    # FAO56 eq. 13
    if is_img(Temp_C):
        temp_add = Temp_C.add(237.3)
        exp_arg = Temp_C.multiply(17.27).divide(temp_add)
        Delta = (exp_arg.exp()).multiply(4098*0.6108).divide(temp_add.pow(2))
        if band_name:
            Delta = Delta.rename(band_name)
    else:
        Delta = 4098*0.6108*np.exp(17.27*Temp_C/(Temp_C + 237.3)) / ((Temp_C + 237.3) ** 2)

    return Delta


def compute_apt_delta_gamma(Temp_C, Press, band_name = None):
    '''
    Function to compute the Priestley-Taylor (PT) term a*delta/(delta + gamma),
    where a is the Priestley-Taylor coefficient.
    Inputs:
        - Temp_C (numpy array or ee.Image): temperature in Celsius.
        - Press (numpy array or ee.Image): the Pressure (Kpa) above sea level.
        Either both inputs are numpy array, or ee.Image, but should not be mixed!
    Outputs: 
        - Apt_Delta_Gamma (numpy array or ee.Image):  the PT term
    References
    ----------
    Priestley and Taylor, 1972        

    Allen et al., 1998
    '''

    import numpy as np

    is_temp_img = is_img(Temp_C)
    is_press_img = is_img(Press)
    
    if is_temp_img != is_press_img:
        print('Either both inputs should be numpy array or ee.Img, but not mixed.')
        return

    Apt = 1.26  #  Priestley-Taylor coefficient
    Gamma = compute_gamma(Press)
    Delta = compute_delta(Temp_C)
    
    if is_temp_img:
        Apt_Delta_Gamma = (Delta.multiply(Apt)).divide(Delta.add(Gamma))
        if band_name:
            Apt_Delta_Gamma = Apt_Delta_Gamma.rename(band_name)
    else:
        Apt_Delta_Gamma = Apt*(Delta)/(Delta + Gamma)

    return Apt_Delta_Gamma


def compute_lai(NDVI, k_par = 0.5, band_name = None):
    '''
    Function to compute the leaf area index (LAI).
    Inputs:
        - NDVI (numpy array or ee.Image): the normalized vegetation index values.
    Outputs: 
        - LAI (numpy array or ee.Image): the leaf area index.
    References
    ----------
    Fisher et al., 2008
    '''

    import numpy as np
    f_c = compute_fipar(NDVI)

    if is_img(f_c):
        log_term = ee.Image(1.0).subtract(f_c)
        k_par = ee.Image(k_par)
        LAI = log_term.log().multiply(-1.0).divide(k_par)
        if band_name:
            LAI = LAI.rename(band_name)
    else:
        LAI = -np.log(1 - f_c)/k_par
    return LAI


def compute_rns(Rn, LAI, band_name=None): 
    '''
    Function to compute the net radiation to the soil (Rns).
    Inputs:
        - Rn (numpy array or ee.Image): the net radiation values (W/m2).
        - LAI (numpy array or ee.Image): the leaf area index (m2/m2).
    Outputs: 
        - Rns (numpy array or ee.Image): the net radiation to the soil (W/m2).
    References
    ----------
    Fisher et al., 2008
    '''

    import numpy as np

    is_Rn_img = is_img(Rn)
    is_LAI_img = is_img(LAI)

    if is_Rn_img != is_LAI_img:
        print('Either both inputs should be numpy array or ee.Img, but not mixed.')
        return

    k_rn = 0.6 

    if is_Rn_img:
        exp_term = LAI.multiply(-k_rn)
        Rns = Rn.multiply(exp_term.exp())
        if band_name:
            Rns = Rns.rename(band_name)
    else:
        Rns = Rn*np.exp(-k_rn*LAI)

    return Rns


def compute_rnc(Rn, Rns, band_name=None):
    '''
    Function to compute the net radiation to the canopy (Rnc).
    Inputs:
        - Rn (numpy array or ee.Image): the net radiation values (W/m2).
        - Rns (numpy array or ee.Image): the net radiation to the soil (W/m2).
    Outputs: 
        - Rnc (numpy array or ee.Image): the net radiation to the canopy (W/m2).
    References
    ----------
    Fisher et al., 2008
    '''

    import numpy as np

    is_Rn_img = is_img(Rn)
    is_Rns_img = is_img(Rns)

    if is_Rn_img != is_Rns_img:
        print('Either both inputs should be numpy array or ee.Img, but not mixed.')
        return

    if is_Rn_img:
        Rnc = Rn.subtract(Rns)
        if band_name:
            Rnc = Rnc.rename(band_name)
    else:
        Rnc = Rn - Rns

    return Rnc


def compute_vpd(RH, Temp_C, band_name=None):
    '''
    Function to compute the vapor pressure deficit in kPa.
    Inputs:
        - RH (numpy array or ee.Image): the relative humidity [0-100].
        - Temp_C (numpy array or ee.Image): array with the temperature
        values in Celsius.         
    Outputs: 
        - VPD (numpy array or ee.Image): the vapor pressure deficit [kPa].
    References
    ----------
    Allen et al., 1998 
    '''

    import numpy as np

    is_RH_img = is_img(RH)
    is_TempC_img = is_img(Temp_C)

    if is_RH_img != is_TempC_img:
        print('Either both inputs should be numpy array or ee.Img, but not mixed.')
        return

    if is_RH_img:
        exp_term = Temp_C.multiply(17.27).divide(Temp_C.add(237.3))
        esat = (exp_term.exp()).multiply(0.6108)
        
        ea = (RH.divide(100.0)).multiply(esat)

        VPD = esat.subtract(ea)
        if band_name:
            VPD = VPD.rename(band_name)
    else:
        # eq. 11
        esat = 0.6108*np.exp(17.27*Temp_C/(Temp_C + 237.3))
    
        # based on eq. 10
        ea = (RH/100.0)*esat
    
        VPD = esat - ea

    return VPD


def compute_fsm(RH, Temp_C, Beta = 1.0, band_name=None):
    '''
    Function to compute the soil moisture constraint Fsm.
    Inputs:
        - RH (numpy array or ee.Image): the relative humidity [0-100].
        - Temp_C (numpy array or ee.Image): array with the temperature
        values in Celsius.  
        - float Beta: sensibility parameter to VPD in kPa.
    Outputs: 
        - Fsm (numpy array or ee.Image): the soil moisture constraint.
    References
    ----------
    Fisher et al., 2008
    '''

    import numpy as np

    is_RH_img = is_img(RH)
    is_TempC_img = is_img(Temp_C)

    if is_RH_img != is_TempC_img:
        print('Either both inputs should be numpy array or ee.Img, but not mixed.')
        return

    VPD = compute_vpd(RH, Temp_C)

    if is_RH_img:
        Fsm = (RH.divide(100)).pow(VPD.divide(Beta))
        Fsm = Fsm.where(Fsm.lt(0), 0)
        Fsm = Fsm.where(Fsm.gt(1), 1)
        if band_name:
            Fsm = Fsm.rename(band_name)
    else:
        #Beta = 1.0 # KPa
        Fsm = (RH / 100.0) ** (VPD/Beta)

        # Bound Fsm
        Fsm = np.array(Fsm)
        Fsm[Fsm < 0] = 0
        Fsm[Fsm > 1] = 1

    return Fsm


def compute_tnoon(Lon, Std_meridian, Doy, band_name = None):
    '''
    Function to compute the solar noon time in decimal hours.
    Inputs:
        - Lon (numpy array or ee.Image): the pixel longitudes.
        - Std_meridian (int or ee.Image): the standard meridian of the study site.
        - int Doy: the day of the year.
    Outputs: 
        - T_noon (numpy array or ee.Image): the solar noon.
    References
    ----------
    Campbell and Norman, 1998
    '''

    import numpy as np

    DTOR = 0.017453292 # constant to convert degrees to radians
    # Compute the value of the equation of time in hours eq (11.4)
    f = (279.575 + 0.9856*Doy)*DTOR
    equation_time = (-104.7*np.sin(f) + 596.2*np.sin(2*f) + 4.3*np.sin(3*f) - 12.7*np.sin(4*f) - 429.3*np.cos(f) - 2.0*np.cos(2*f) + 19.3*np.cos(3*f))/3600.0
 
    if is_img(Lon):
        LC = (Lon.subtract(Std_meridian)).multiply(4).divide(60)
        T_noon = ee.Image(12.0).subtract(LC).subtract(equation_time)
        if band_name:
            T_noon = T_noon.rename(band_name)
    else:
        # Compute the time of solar noon eq (11.3)
        LC = (4*(Lon - Std_meridian))/60.0 # compute latitude correction in hours
        T_noon = 12.0 - LC - equation_time
    
    return T_noon


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


def ptjpl_arid(RH, Temp_C, Press, Rn, NDVI, F_aparmax, LAI=None, G=None,
G_Params = [0.31, 74000, 10800], eot_params=None, k_par=0.5, Beta=1.0, Mask=1.0, Mask_Val=0.0, band_names=['LE', 'LEc', 'LEs', 'LEi', 'H', 'G', 'Rn']):
    '''
    Function to compute evapotranspiration components using the PT-JPL model
    adapted for arid lands (Aragon et al., 2018)

    Inputs:
        - RH (numpy array or ee.Image): relative humidity [0-100].
        - Temp_C (numpy array or ee.Image): air temperature in Celsius.
        - Press (numpy array or ee.Image): Pressure in Kpa.
        - Rn (numpy array or ee.Image): Net radiation in W/m2.
        - NDVI (numpy array or ee.Image): NDVI values.
        - F_aparmax (numpy array or ee.Image): Maximum fapar values.
    Optional_Inputs:
        - LAI (numpy array or ee.Image): Leaf area index (m2/m2). 
                If not provided, LAI is computed (see compute_lai).
        - G (numpy array or ee.Image): Soil heat flux (W/m2).
                If not provided, G is computed (see compute_g).
        - G_params (list [float A, float B, float C]): the parameters for
          computing solid heat flux where A is the maximum ratio of G/Rns
          B reduces deviation of G/Rn to measured values, also thought of
          as the spread of the cosine wave and C is the phase shift between
          the peaks of Rns and G. B and C are in seconds.
        - eot_params (list [int Doy, float Time, float longitude, float stdMerid]):
            Parameters for the equation of time. Required if G is not supplied 
            (see compute_tnoon and compute_g)
            If the inputs are ee.Images, longitude and stdMerid are not required and
            they are ignored.
        - Doy (integer): the day of year. Required if G is not supplied (see compute_g)
        - Time (float): the decimal time of the computation. Required if G is not supplied (see compute_g) 
        - k_par (float): parameter used in the computation of LAI (see compute_lai) 
        - Beta (float): Sensibility parameter to VPD in kPa (see compute_fsm)
        - Mask (numpy array): an array containing the coded invalid data (e.g. -9999)
            of the same size as the input image. Only for numpy array inputs; ignored for
            ee.Image inputs.
        - Mask_Val (number): the value that represents invalid data. Only for numpy array
            inputs; ignored for ee.Image inputs. 
        - band_names (list of strings): If provided, the bands in the output ee.Image
                are renamed using this list. Ignored if inputs are numpy arrays.  
    Outputs: 
        - ET (tuple or ee.Image): tuple containing numpy arrays with the following
                                  components, or ee.Image containing the following bands:
        -     LE: the latent heat flux.
        -     LEc: the canopy transpiration component of LE.
        -     LEs: the soil evaporation component of LE.
        -     LEi: the interception evaporation component of LE.
        -     H: the sensible heat flux.
        -     G: the sensible heat flux.
        -     Rn: the net radiation.
    '''
    import numpy as np
    is_RH_img = is_img(RH)
    is_Temp_C_img = is_img(Temp_C)
    is_Press_img = is_img(Press)
    is_Rn_img = is_img(Rn)
    is_NDVI_img = is_img(NDVI)
    is_F_aparmax_img = is_img(F_aparmax)
    all_conditions = [is_RH_img, is_Temp_C_img, is_Press_img, is_Rn_img, 
    is_NDVI_img, is_F_aparmax_img]
 
    if LAI:
        is_LAI_img = is_img(LAI)
        all_conditions = all_conditions.append(is_LAI_img)
    if G:
        is_G_img = is_img(G) 
        all_conditions = all_conditions.append(is_G_img)  
 
    
    # Either all inputs are images, or not:
    test_all = np.unique(all_conditions)    
    if(len(test_all)>1):
        print('Either all inputs should be ee.Images, or all inputs should be numeric.')
        return

    are_inputs_imgs = test_all[0]

    fwet = compute_fwet(RH) 
    fg = compute_fg(NDVI)
    ft = compute_ft_arid(Temp_C)
    f_apar = compute_fapar(NDVI)
    fm = compute_fm(f_apar, F_aparmax)
    taylor = compute_apt_delta_gamma(Temp_C, Press)

    if not LAI:
        LAI = compute_lai(NDVI, k_par)
    
    rns = compute_rns(Rn, LAI)
    rnc = compute_rnc(Rn, rns)
    fsm = compute_fsm(RH, Temp_C, Beta)

    if not G:
        # Check that equation of time parameters are supplied
        # these are: Doy, Time, longitude, and standard meridian
        # if the inputs are ee.Image, longitude and standard meridian are not required
        if are_inputs_imgs:
            if len(eot_params)>=2: 
                Doy=eot_params[0]
                Time=eot_params[1]
                longitude = ee.Image.pixelLonLat().select('longitude')
                stdMerid = longitude.add(187.5).divide(15).int().multiply(15).subtract(180)
                t_noon = compute_tnoon(longitude, stdMerid, Doy)
                G = compute_g(Time, t_noon, rns, G_Params=G_Params)
            else:
                print('Equation of time parameters (list) [Doy, Time] are required to compute G.')
                return
        else:
            if len(eot_params)==4:
                Doy=eot_params[0]
                Time=eot_params[1]
                longitude=eot_params[2]
                stdMerid=eot_params[3]
                t_noon = compute_tnoon(longitude, stdMerid, Doy)
                G = compute_g(Time, t_noon, rns, G_Params = G_Params)
            else:
                print('Equation of time parameters (list) [Doy, Time, longitude, standard meridian] are required to compute G.')
                return

    # Compute ET components:
    if are_inputs_imgs:
        cst_1 = ee.Image(1.0)
        fwet_sub1 = cst_1.subtract(fwet)
        # Canopy transpiration image
        LEc = fwet_sub1.multiply(fg).multiply(ft).multiply(fm).multiply(taylor).multiply(rnc)
        # Soil evaporation image
        LEs = ((fwet_sub1.multiply(fsm)).add(fwet)).multiply(taylor).multiply(rns.subtract(G))
        # Interception evaporation image
        LEi = fwet.multiply(taylor).multiply(rnc)
        # Evapotranspiration image
        LE = LEc.add(LEs).add(LEi)
        # Compute the sensible heat flux image H by residual
        H = (Rn.subtract(G)).subtract(LE)
        Rn = rns.add(rnc)

        # Prepare the output image:
        # LE, LEc, LEs, LEi, H, G, Rn
        ET = LE.addBands(LEc).addBands(LEs).addBands(LEi).addBands(H).addBands(G).addBands(Rn)
        ET = ET.rename(band_names)
    else:
        # Compute canopy transpiration
        LEc = (1 - fwet)*fg*ft*fm*taylor*rnc
        # Compute soil evaporation
        LEs = (fwet + fsm*(1 - fwet)) * taylor*(rns - G)
        # Compute interception evaporation
        LEi = fwet*taylor*rnc
        # Compute the evapotranspiration
        LE = LEc + LEs + LEi
        # Compute the sensible heat flux H by residual
        H = (Rn - G) - LE
        Rn = rns + rnc
        
        Mask = np.array(Mask).astype(float)
        Mask[Mask != Mask_Val] = 1.0
        Mask[Mask == Mask_Val] = np.nan    

        # Mask ensures proper fluxes and size of the outputs
        LE = LE * Mask
        LEc = LEc * Mask
        LEs = LEs * Mask
        LEi = LEi * Mask
        H = H * Mask
        G = G * Mask
        Rn = Rn * Mask
        LAI = LAI * Mask
        
        # output as np.nstack:
        ET=np.dstack((LE, LEc, LEs, LEi, H, G, Rn))

    return ET


main_ref="Aragon, B., et al. (2018). \"CubeSats Enable High Spatiotemporal Retrievals \
of Crop-Water Use for Precision Agriculture\". Remote Sensing 10(12): 1867."
# Citation:
# cite() - main reference only
def cite():
    print(main_ref)
# cite_all() - all references
def cite_all():
    for ref in all_refs:
        print(ref)

all_refs=["Allen, R.G., Pereira, L.S., Raes, D., Smith, M. \
\"Crop evapotranspiration —guidelines for computing crop water requirements\" \
(1998) FAO Irrigation and drainage paper 56. Food and Agriculture \
Organization, Rome, pp. 35-39. \
http://www.fao.org/docrep/x0490e/x0490e00.htm", 
"Aragon, B., Houborg, R., Tu, K., Fisher, J.B., McCabe, M. \
\"Cubesats enable high spatiotemporal retrievals of crop-water use for \
precision agriculture (2018)\" \
Remote Sensing, 10 (12), art. no. 1867. \
http://dx.doi.org/10.3390/rs10121867",
"Campbell, G. S., & Norman, J. M. \
\"Introduction to environmental biophysics (2nd ed.) (1998)\"\
New York: Springer, pp. 168-169\
http://dx.doi.org/10.1007/978-1-4612-1626-1",
"Carlson, T.N., Capehart, W.J., Gillies, R.R. \
\"A new look at the simplified method for remote sensing of daily \
evapotranspiration (1995)\" \
Remote Sensing of Environment, 54 (2), pp. 161-167.\
http://dx.doi.org/10.1016/0034-4257(95)00139-R",
"Choudhury, B.J., Ahmed, N.U., Idso, S.B., Reginato, R.J., Daughtry, C.S.T. \
\"Relations between evaporation coefficients and vegetation indices \
studied by model simulations (1994)\" \
Remote Sensing of Environment, 50 (1), pp. 1-17.\
http://dx.doi.org/10.1016/0034-4257(94)90090-6",
"Fisher, J.B., Tu, K.P., Baldocchi, D.D. \
\"Global estimates of the land-atmosphere water flux based on monthly \
AVHRR and ISLSCP-II data, validated at 16 FLUXNET sites (2008)\" \
Remote Sensing of Environment, 112 (3), pp. 901-919.\
http://dx.doi.org/10.1016/j.rse.2007.06.025",
"Potter, C.S., Randerson, J.T., Field, C.B., Matson, P.A., Vitousek, P.M.,\
Mooney, H.A., Klooster, S.A. \
\"Terrestrial ecosystem production: A process model based on global \
satellite and surface data (1993)\" \
Global Biogeochemical Cycles, 7 (4), pp. 811-841. \
http://dx.doi.org/10.1029/93GB02725",
"Priestley, C.H.B. and Taylor, R.J. \
\"On the Assessment of Surface Heat Flux and Evaporation Using Large Scale \
Parameters (1972)\" Monthly Weather Review, 100, 81-92. \
http://dx.doi.org/10.1175/1520-0493(1972)100<0081:OTAOSH>2.3.CO;2",
"Joseph A. Santanello Jr. and Mark A. Friedl. \
\"Diurnal Covariation in Soil Heat Flux and Net Radiation (2003)\" \
J. Appl. Meteor., 42, pp. 851-862. \
Remote Sensing of Environment, 112 (3), pp. 901-919. \
http://dx.doi.org/10.1175/1520-0450(2003)042<0851:DCISHF>2.0.CO;2"] 