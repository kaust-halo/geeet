"""Functions related to vegetation/crop modeling
fapar, fipar, vegetation cover, LAI, etc. and other biophysical parameters
(green canopy fraction, plant temperature and moisture constraints, soil moisture constraint)
"""
from geeet.common import is_eenum, is_img
import numpy as np

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

    if is_img(NDVI):
        from ee import Image
        # if NDVIsoil or NDVIveg are float, convert to ee.Image:
        if not is_img(NDVIsoil):
            NDVIsoil = Image(NDVIsoil)
        
        if not is_img(NDVIveg):
            NDVIveg = Image(NDVIveg)
 
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
    Compute the fraction of the photosynthetic 
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


    m2 = 1.0
    b2 = -0.05

    if is_img(NDVI):
        from ee import Image
        m2 = Image(m2)
        b2 = Image(b2)
        F_ipar = (m2.multiply(NDVI)).add(b2)
        F_ipar = F_ipar.where(F_ipar.lt(0), 0)
        if band_name:
            F_ipar = F_ipar.rename(band_name)
    else:
        F_ipar = m2*NDVI + b2
        F_ipar = np.array(F_ipar)
        F_ipar[F_ipar < 0] = 0

    return F_ipar


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

    f_c = compute_fipar(NDVI)

    if is_img(f_c):
        from ee import Image
        log_term = Image(1.0).subtract(f_c)
        k_par = Image(k_par)
        LAI = log_term.log().multiply(-1.0).divide(k_par)
        if band_name:
            LAI = LAI.rename(band_name)
    else:
        LAI = -np.log(1 - f_c)/k_par
        LAI = np.array(LAI)
    return LAI


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


def compute_ft_arid(T_a, band_name=None):
    '''
    Function to compute the plant temperature constraint Ft.
    This function is specific for arid environments
    characterized by non-limiting air temperatures. Here ft only constrains 
    low temperature values. 
    ft = 1 / (1 + exp(0.2*(12-T))) (equation 12 in Aragon et al., 2018,
    where T is in Celsius)
    Inputs:
        - T_a (numpy array or ee.Image): the air temperature [K].
    Outputs: 
        - Ft (numpy array or ee.Image): the plant temperature constraint. 
    References
    ----------
    Potter et al., 1993        

    Aragon et al., 2018        
    '''

    if is_img(T_a):
        from ee import Image
        cst1 = Image(1.0)
        exp_arg = (Image(12.0).subtract(T_a.subtract(273.15))).multiply(0.2)
        Ft = cst1.divide(cst1.add(exp_arg.exp()))
        if band_name:
            Ft = Ft.rename(band_name)
    else:
        Ft = 1.0 /(1.0 + np.exp(0.2*(12 - T_a + 273.15)))
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


def compute_fsm(RH, Temp_K, Beta = 1.0, band_name=None):
    '''
    Function to compute the soil moisture constraint Fsm (original).
    Inputs:
        - RH (numpy array or ee.Image): the relative humidity [0-100].
        - Temp_K (numpy array or ee.Image): array with the temperature
        values in Kelvin.  
        - float Beta: sensibility parameter to VPD in kPa.
    Outputs: 
        - Fsm (numpy array or ee.Image): the soil moisture constraint.
    References
    ----------
    Fisher et al., 2008
    '''

    from geeet.meteo import vpd as compute_vpd

    is_RH_img = is_img(RH)
    is_TempK_img = is_img(Temp_K)

    if is_RH_img != is_TempK_img:
        print('Either both inputs should be numpy array or ee.Img, but not mixed.')
        return

    VPD = compute_vpd(RH, Temp_K)  # in KPa

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

def compute_ftheta(LAI, theta = 0, band_name = 'f_theta'):
    """
    Compute the fraction of field of view of the infrared radiometer occupied by canopy
    (fractional vegetation cover within the sensor field of view)
    F(theta) = 1-exp(-0.5LAI/cos(theta))  (Eqn 2. from Norman et al., 1995)
    
    Inputs:
        - LAI (numpy array or ee.Image): leaf area index (m2/m2)
        - theta (scalar, float): view angle from sensor (e.g. for Landsat it is 0), in degrees.
    Outputs: 
        - f_theta (numpy array or ee.Image): the fraction of field of view occupied by canopy.
    References
    ----------
    Norman et al., 1995 Eq 2
    Anderson et al., 1997 Eq 4
    """
    import numpy as np
    DTOR = np.pi/180.0 # constant to convert degrees to radians

    if is_img(LAI):
        from ee import Image
        
        if is_eenum(theta):
            cos_theta = (theta.multiply(DTOR)).cos() 
        else:
            cos_theta = np.cos(np.radians(theta))
        f_theta = Image(1.0).subtract((LAI.multiply(-0.5).divide(cos_theta)).exp())
        f_theta = f_theta.where(f_theta.gt(0.9), 0.9) 
        f_theta = f_theta.where(f_theta.lt(0.05), 0.05)
        f_theta = f_theta.rename(band_name)
    else:
        f_theta = 1 - np.exp(-0.5*LAI/np.cos(np.radians(theta))) 
        f_theta = np.array(f_theta)
        f_theta[f_theta > 0.9] = 0.9
        f_theta[f_theta < 0.05] = 0.05
    return f_theta

def compute_Rns(Rn, LAI, solar_angles=None, use_zenith = False, k=0.6, LAI_thre = None):
    """
    Compute the soil net radiation as:
    Rns = Rn exp(-k*LAI)  
    or 
    Rns = Rn exp (-K*LAI/sqrt(2*cos(theta_z)))
    
    In PT-JPL the first form is used (use_angle = False), with a recommended
    value (for irrigated arid regions) of k = 0.6 (Aragon et al., 2018)

    In the original TSEB (Norman et al., 1995), the first form is used
    (see equation 13): Rnsoil = Rn exp(0.9 ln (1-fc)) where fc = 1-exp(-0.5LAI)
    i.e. Rnsoil = Rn exp(0.9(-0.5LAI))  i.e. k = 0.45

    In TSEB, the solar zenith angle (theta_z) is considered.
    If LAI_thre is provided, separate recommended values for sparse and dense vegetation is used
    based on Zhuang and Wu, 2015, i.e.:
    low LAI (LAI < LAI_thre): k=0.45
    high LAI (LAI >= LAI_thre): k=0.18
    Otherwise the k 

    for high LAI values and K=0.18 for low LAI values (Zhuang and Wu, 2015).
    This option is used if LAI_thre is specified (e.g. LAI_thre = 2) (specified k is therefore
    ignored). 

    Canopy radiation can then be obtained as Rnc = Rn - Rns

    Inputs:
        - Rn (numpy array or ee.Image): Net radiation in the surface (W/m2) (see geeet.solar.compute_Rn)
        - LAI (numpy array or ee.Image): Leaf Area Index (m2/m2)
        - solar_angles (tuple or ee.Image): Solar zenith and azimuth angles (degrees)
                                            Required only if use_angle = True (e.g. for TSEB)
                                            (only zenith is used here). See geeet.solar.compute_solar_angles
                                            If tuple, the first element is the zenith, i.e.: (zenith, azimuth)
                                            If ee.Image, the zenith angle should be available as a 
                                            band named "zenith".
        - use_angle (boolean): Whether to use the first or second form of Rns
        - k (float): parameter to scale LAI (see equations above).  
    Optional:
        - LAI_thre (float): Separate low (LAI<LAI_thre) and high (LAI>=LAI_thre) LAI values 
                            and use a separate k as recommended by Zhuang and Wu, 2015.
                            If LAI_thre is given, then the parametr k is ignored and 
                            the following is used: k(LAI<LAI_thre) = 0.8; k(LAI>=LAI_thre) =0.45 

    Outputs: 
        - Rns (numpy array or ee.Image): Soil net radiation (W/m2).

    References:
    ----------
    Zhuang and Wu, 2015    
    Fisher et al., 2008
    Aragon et al., 2018
    """
    DTORAD = np.pi/180
    is_Rn_img = is_img(Rn)
    is_LAI_img = is_img(LAI)

    if is_Rn_img != is_LAI_img:
        print('Either both inputs should be numpy array or ee.Img, but not mixed.')
        return

    if is_Rn_img:
        from ee import Image
        k = Image(k)
        if LAI_thre is not None:
            # Use separate recommended k's for partial vs dense canopy cover:
            k = k.where(LAI.gte(LAI_thre), 0.45)
            k = k.where(LAI.lt(LAI_thre), 0.8)
    
        exp_term = LAI.multiply(k).multiply(-1)
        if use_zenith:
            zenith = solar_angles.select('zenith')
            angle_term = (zenith.multiply(DTORAD)).cos().multiply(2).sqrt()
            exp_term = exp_term.divide(angle_term)
        Rns = Rn.multiply(exp_term.exp())
    else:
        if LAI_thre is not None:
            # If LAI threshold is specified, use the recommended values
            # from Zhuang and Wu, 2015 (k supplied is ignored):
            k = np.ones_like(LAI)
            k = np.where(LAI<LAI_thre, 0.8, 0.45)

        exp_term = -k*LAI
        if use_zenith:
            zenith, _ = solar_angles
            exp_term = exp_term/(np.sqrt(2*np.cos(zenith*DTORAD)))
        Rns = Rn*np.exp(exp_term)
        Rns = np.array(Rns)
    return Rns