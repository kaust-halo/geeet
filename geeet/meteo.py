"""Meteorological functions"""
from geeet.common import is_img
import numpy as np
# Constants defined as in 
# https://www.ecmwf.int/sites/default/files/elibrary/2016/17117-part-iv-physical-processes.pdf#subsection.L.5
Rdry = 287.0597 # gas constant for dry air, J/(kg*degK)
Rvap = 461.5250 # gas constant for water vapor, J/(kg*degK)
epsilon = Rdry/Rvap # (~0.622) ratio of the molecular weight of water vapor to dry air
c_pd = (7/2)*Rdry # Heat capacity of dry air at constant pressure, J kg-1 K-1 
c_pv = 4*Rvap # Heat capacity of water vapor at constant pressure, J kg-1 K-1
# Constants for Teten's formula using parameters from Buck (1981)
# for saturation over water. 
a1 = 611.21 # in Pa
a3 = 17.502 
a4 = 32.19 # in K
T0 = 273.16 # in K
 
def teten(T):
    '''
    Compute Teten's formula for saturation water vapour pressure  (esat (T)) in Pa 
    with parameters set according to Buck (1981) for saturation over water. 
    Reference: 
    https://www.ecmwf.int/sites/default/files/elibrary/2016/17117-part-iv-physical-processes.pdf#subsection.7.2.1    

    Input: T (numpy array or ee.Image) Temperature in Kelvin
    '''
    if is_img(T):
        T1 = T.subtract(T0) # in K
        T2 = T.subtract(a4) # in K
        esat = T1.divide(T2).multiply(a3).exp().multiply(a1)
    else:
        esat = a1*np.exp(a3*(T-T0)/(T-a4))
    return esat

def specific_humidity(T,P):
    '''
    Input: (ee.Images or np.arrays):
        - P: surface pressure in Pascals
        - T: temperature in Kelvin 
    Output: (ee.Image or np.array):
        - Q: specific humidity 
    '''
    if is_img(T):
        esat = teten(T)
        denom = P.subtract(esat.multiply(1-epsilon))
        Q = esat.multiply(epsilon).divide(denom)
    else:
        esat = teten(T)
        Q = epsilon*esat/(P-(1-epsilon)*esat)
    return Q

def relative_humidity(temperature, dewpoint_temperature, pressure, band_name='relative_humidity'):
    '''
    Input: (ee.Images or np.arrays):
        - temperature, in Kelvin
        - dewpoint_temperature, in Kelvin
        - pressure: surface pressure in Pascals
    Output: (ee.Image or np.array):
        - RH: relative humidity(%)
    
    Equation 7.91 in:
    https://www.ecmwf.int/sites/default/files/elibrary/2016/17117-part-iv-physical-processes.pdf
    '''
    if is_img(temperature):
        Q = specific_humidity(dewpoint_temperature,pressure)
        esat = teten(temperature)
        denom = Q.multiply(1/epsilon -1).add(1).multiply(esat)
        RH = pressure.multiply(Q).multiply(100/epsilon).divide(denom).rename(band_name)
    else:
        esat = teten(temperature)
        Q = specific_humidity(dewpoint_temperature,pressure)
        RH = (pressure*Q*100/epsilon)/(esat*(1+Q*((1/epsilon) - 1)))
    return RH

def vpd(RH, Temp_K, band_name=None):
    '''
    Function to compute the vapor pressure deficit in kPa.
    Inputs:
        - RH (numpy array or ee.Image): the relative humidity [0-100].
        - Temp_K (numpy array or ee.Image): array with the temperature
        values in Kelvin.         
    Outputs: 
        - VPD (numpy array or ee.Image): the vapor pressure deficit [kPa].
    References
    ----------
    Allen et al., 1998 
    '''
    is_RH_img = is_img(RH)
    is_TempK_img = is_img(Temp_K)

    if is_RH_img != is_TempK_img:
        print('Either both inputs should be numpy array or ee.Img, but not mixed.')
        return

    esat = teten(Temp_K) # in Pa

    if is_RH_img:
        ea = (RH.divide(100.0)).multiply(esat)
        VPD = esat.subtract(ea)
        VPD = VPD.divide(1000) # convert to KPa
        if band_name:
            VPD = VPD.rename(band_name)
    else:
        ea = (RH/100.0)*esat
        VPD = esat - ea
        VPD = VPD/1000.0 # conver to KPa
    return VPD


def LatHeatVap(Temp_K):
    """ Calculates the Latent heat of vaporization
    
    Inputs (ee.Image or np.array):
    - temperature: air temperature (Kelvin).
    Outputs (ee. Image or np.array):
    - L: latent heat of vaporization (MJ kg-1)

    based on Eq. 3-1 Allen FAO98
    """
    if is_img(Temp_K):
        from ee import Image
        L = Image(2.501).subtract((Temp_K.subtract(273.15)).multiply(2.361e-3))
    else:
        L = 2.501 - (2.361e-3*(Temp_K-273.15)) # at 20C this is ~2.45 MJ kg-1
    return L

def compute_met_params(temperature, pressure):
    """
    Calculates several temperature and/or pressure-dependent
    parameters related to heat flux in air,
    which are commonly used in ET models

    Inputs (ee.Image or np.arrays):
    - temperature: air temperature at reference height (Kelvin).
    - pressure: total air pressure (dry air + water vapor) (Pa)
    
    Outputs: (ee.Image with following bands, OR list of np.arrays:)
    - q (specific humidity)
    - ea (water vapor pressure), in Pa
    - rho (air density), in kg m-3
    - cp (air heat capacity), in (J kg-1 K-1)
    - s (delta vapor pressure, i.e. slope of the saturation water vapor pressure), in Pa K-1
    - lambda (latent heat of vaporization), in MJ kg-1
    - psicr (psicrometric constant), in Pa K-1
    - taylor (=s/(s+psicr)), in Pa K-1
    """
    q = specific_humidity(temperature, pressure) 
    ea = teten(temperature)  # in Pa
    Lambda = LatHeatVap(temperature)  # in MJ kg-1

    if is_img(pressure):
        from ee import Image
        mfactor = Image(1.0).subtract(ea.multiply(1.0-epsilon).divide(pressure))
        rho = pressure.divide(temperature.multiply(Rdry)).multiply(mfactor)
        cp = ((Image(1.0).subtract(q)).multiply(c_pd)).add(q.multiply(c_pv))
        s = ea.multiply(a3*(T0-a4)).divide((temperature.subtract(a4)).pow(2))
        psicr = cp.multiply(pressure).divide(Lambda.multiply(epsilon*1e6))
        taylor = s.divide(s.add(psicr))
        met_params = q.addBands(ea).addBands(rho).addBands(cp).addBands(s).addBands(Lambda).addBands(psicr).addBands(taylor)
        met_params = met_params.rename(['q', 'ea', 'rho', 'cp', 'delta', 'Lambda', 'gamma', 'taylor'])
    else:
        # Hydrology - An Introduction (Brutsaert 2005) eq 2.6 (pp 25)
        rho = (pressure/(Rdry * temperature )) * (1.0 - (1.0 - epsilon) * ea / pressure)
        # Rearranged from https://www.ecmwf.int/sites/default/files/elibrary/2016/17117-part-iv-physical-processes.pdf#section.2.7
        cp = (1.0-q)*c_pd + q*c_pv
        # Slope of saturation water vapor pressure (i.e. slope of teten's formula):
        #esat = a1*np.exp(a3*(T-T0)/(T-a4))
        # desat/dT = 
        # a3*(T0-a4) *   a1*np.exp(a3*(T-T0)/(T-a4))    /(T-a4)**2
        # = a3*(t0-a4) * esat / (T-a4)**2
        s = a3*(T0-a4)*ea/((temperature-a4)**2)  # in Pa K-1
        # Psicrometric constant 
        psicr = cp*pressure/(epsilon*Lambda*1e6)  # Pa/K  
        # Priestley-Taylor term DELTA/ (DELTA+GAMMA)
        taylor = s/(s+psicr) 

        met_params = [np.array(q), np.array(ea), np.array(rho), np.array(cp), np.array(s), np.array(Lambda), np.array(psicr), np.array(taylor)]
    return met_params

def compute_roughness(CH, fm=0.125, fd=0.65, kb=2.0, min_values = [0.003, 0.003, 0.004], band_names = ['ZM', 'ZH', 'D0']):
    """
    Roughness length (m) for momentum and heat transport (ZM, ZH)
    and zero-plane displacement height (m) (D0)

    Inputs: 
    - CH: canopy height in m (ee.Image or numpy array)
    Scalar (optional) inputs:
    - fm: ratio of vegetation height used for ZM (default is 0.125)
    - fd: ratio of vegetation height used for D0 (default is 0.65)
    - kb: parameter kb=ln(ZM/ZH) (default is 2.0)
    - min_values: minimum values for ZM, ZH, and D0 given as a list. 
    - band_names: if provided, rename the output ee.Image with these names
                  Defaults to 'ZM', 'ZH', and 'D0'

    Outputs: rough (ee.Image or list) containing ZM, ZH, and D0 
                either as bands in an ee.Image or numpy arrays    
    
    ZM and D0 are based on simple fractions (fm, fd) of canopy height
    while ZH is based on a log relation between ZM and ZH:  ln(ZM/ZH)~kb
    these are all mentioned in-text in Norman et al., 1995 (N95):
    ZM = canopy height * fm  Brutsaert (1982) 
    ZH = ZM/exp(kb)          Garrat and Hicks (1973) mentioned in N95
    D0 = canopy height * fd  Brutsaert (1982)

    The default ratio values for ZM and D0 are fm=1/8 and fd=0.65, respectively 
    The default kb parameter is 2.0
    Minimum values for all parameters can be set by default (3 mm, 3 mm, and 4 mm).
    """
    import numpy as np
    zM_min, zH_min, D0_min = min_values

    if is_img(CH):
        from ee import Image
        ZM = CH.multiply(fm)
        ZM = ZM.max(Image(zM_min))

        ZH = ZM.divide(Image(kb).exp())
        ZH = ZH.max(Image(zH_min))

        D0 =  CH.multiply(fd)
        D0 = D0.max(Image(D0_min))
        rough = ZM.addBands(ZH).addBands(D0).rename(band_names)
    else:
        ZM = np.maximum(zM_min, fm*CH)
        ZH = np.maximum(zH_min, ZM/np.exp(kb))
        D0 = np.maximum(D0_min, fd*CH)
        rough = [ZM, ZH, D0]
    return rough 