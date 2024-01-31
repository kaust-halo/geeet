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
        F_ipar = (
            m2*NDVI + b2  
        ).clip(0,1)

        if hasattr(F_ipar, "rename"):
            F_ipar = F_ipar.rename("fipar")

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
        if hasattr(LAI, "rename"):
            LAI = LAI.rename("LAI")

    return LAI

def lai_houborg2018(blue = None, green = None, red = None, nir = None, 
    swir1 = None, swir2 = None, additional_models = False, band_name = 'LAI'):
    '''
    Hard-coded Cubist-based LAI model from a hybrid training approach
    described in Houborg and McCabe (2018), ISPRS J. Photogramm., 135, 173-188 
    (https://doi.org/10.1016/j.isprsjprs.2017.10.004)

    The model was trained using data from a desert environment region with 
    irrigated agricultural fields (Al Kharj, Saudi Arabia). The data included 
    in-situ measurements (n=87) and from a physically-based model (REGFLEC; n=1713). 
    Applicable only for this region using Landsat-8 surface reflectance data. 
 
    Default model (equation 1 in Houborg and McCabe, 2018):
    Rule 1 (MSR <= 1.1384):
    LAI = -1.852 - 0.456 SR + 2.45 NDVI + 15.87 NDVI^2 + 0.8 EVI2 + 31.56 EVI2^2
        - 3.64 OSAVI - 36.35 OSAVI^2 + 1.5 EVI - 3.3 EVI^2 + 3.8 MSR
        - 4.38 NDWI - 5.96 NDWI^2 - 0.38 NDWI2 + 0.27 NDWI2^2
    
    Rule 2 (MSR > 1.1384):
    LAI = 3.061 - 0.004 SR + 87.32 NDVI - 19.65 NDVI^2 + 94.43 EVI2 - 29.5 EVI2^2
        - 172.66 OSAVI + 38.7 OSAVI^2 - 0.87 EVI - 2.95 EVI^2 + 3.45 MSR
        - 5.34 NDWI - 2.35 NDWI^2 - 18.08 NDWI2 + 14.54 NDWI2^2  

    Inputs: 
    - blue (numpy array or ee.Image): surface reflectance blue band
    - red (numpy array or ee.Image): surface reflectance red band
    - nir (numpy array or ee.Image): surface reflectance near-infrared band
    - swir1 (numpy array or ee.Image): surface reflectance short-wave infrared band 1
    - swir2 (numpy array or ee.Image): surface reflectance short-wave infrared band 2

    Optional inputs:
    - additional_models (boolean): If true, it computes two additional cubist models
        (See below). These additional models don't appear on the paper. They require
        that the green band is also provided. 
    - green (numpy array or ee.Image): surface reflectance green band. Required only if
        additional_models is True. 
    - band_name (str): rename the output using band_name

    Outputs:
    - lai (numpy array or ee.Image): leaf area index

    If "additional_models" is true, LAI will be based on a combination
    of three models, specifically:
    
    0.4*(default model) + 0.3*(additional model 1) + 0.3*(additional model 2)
    
    Additional model 1:
    Rule 1 (MSR <= 1.13838):
    LAI = -2.9213822 - 25.32 OSAVI^2 + 21.57 EVI2^2 + 5.35 MSR + 9.93 NDVI^2
          - 7.01 OSAVI - 0.342 SR - 8.6 NDWI^2 - 6.15 NDWI + 4.22 NDVI
          + 3.92 EVI2 - 0.53 NDWI2 + 0.48 NDWI2^2

    Rule 2 (MSR > 1.13838):
    LAI = 3.5351541 - 173.11 OSAVI + 88.72 NDVI + 94.29 EVI2 + 33.64 OSAVI^2
        - 31.69 EVI2^2 - 20.64 NDWI2 - 18.62 NDVI^2 + 17.36 NDWI2^2
        + 3.31 MSR - 5.32 NDWI - 2.23 NDWI^2 - 0.003 SR

    Additional Model 2
    Rule 1 (GSR <= 2.41101):
    LAI = -1.1382301 + 1.826 SR - 28.13 OSAVI^2 + 29.04 EVI2^2 - 12.68 OSAVI
          - 0.719 GSR + 5.52 EVI2 + 4.73 NDVI + 2.97 GNDVI^2 + 1.03 GNDVI
          - 0.31 MTVI2 - 0.41 MTVI2^2

    Rule 2 (GSR > 2.41101):
    LAI = 0.028714 - 117.51 OSAVI + 66.55 EVI2 + 44.51 NDVI + 13.99 NDVI^2
      - 12.33 GNDVI^2 + 8.01 MTVI2 + 0.968 GSR - 7.91 OSAVI^2
      - 8.07 EVI2^2 - 6.26 MTVI2^2 + 5.1 GNDVI - 0.052 SR
    

    References
    ----------
    Houborg and McCabe (2018)
    + references appearing in Table 3
    '''
    if is_img(blue):
        # Compute vegetation indices used in this model
        # EVI, EVI2, MSR, NDVI, NDWI, NDWI2, OSAVI, SR
        # as defined in Houborg et al., 2018

        # Enhanced VI (EVI):
        EVI = (nir.subtract(red).multiply(2.5)
              .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)))
        EVIp2 = EVI.pow(2)
        # 2-band EVI:
        EVI2 = ((nir.subtract(red)).multiply(2.5)
            .divide(nir.add(red.multiply(2.4)).add(1)))
        EVI2p2 = EVI2.pow(2)
        # Mid-infrared simple ratio (MSR):
        MSR = nir.divide(swir1)
        # Normalized difference vegetation index (NDVI):
        NDVI = (nir.subtract(red)).divide(nir.add(red))
        NDVIp2 = NDVI.pow(2)
        # Normalized difference water index (NDWI):
        NDWI = (nir.subtract(swir1)).divide(nir.add(swir1)).add(0.24)
        NDWIp2 = NDWI.pow(2)
        # Normalized difference water index 2 (NDWI2):
        NDWI2 = (nir.subtract(swir2)).divide(nir.add(swir2)).add(0.22)
        NDWI2p2 = NDWI2.pow(2)
        # Optimized soil adjusted vegetation index (OSAVI):
        OSAVI = ((nir.subtract(red)).multiply(1.16)
            .divide(nir.add(red).add(0.16)))
        OSAVIp2 = OSAVI.pow(2)
        # Simple ratio:
        SR = nir.divide(red)

        # rules:
        lai1 = ((EVI.multiply(1.5)).subtract(EVIp2.multiply(3.3))
            .add(EVI2.multiply(0.8)).add(EVI2p2.multiply(31.56))
            .add(MSR.multiply(3.8))
            .add(NDVI.multiply(2.45)).add(NDVIp2.multiply(15.87))
            .subtract(NDWI.multiply(4.38)).subtract(NDWIp2.multiply(5.96))
            .subtract(NDWI2.multiply(0.38)).add(NDWI2p2.multiply(0.27))
            .subtract(OSAVI.multiply(3.64)).subtract(OSAVIp2.multiply(36.35))
            .subtract(SR.multiply(0.456)).subtract(1.852))

            
        lai2 = ((EVI.multiply(-0.87)).subtract(EVIp2.multiply(2.95))
            .add(EVI2.multiply(94.43)).subtract(EVI2p2.multiply(29.5))
            .add(MSR.multiply(3.45))
            .add(NDVI.multiply(87.32)).subtract(NDVIp2.multiply(19.65))
            .subtract(NDWI.multiply(5.34)).subtract(NDWIp2.multiply(2.35))
            .subtract(NDWI2.multiply(18.08)).add(NDWI2p2.multiply(14.54))
            .subtract(OSAVI.multiply(172.66)).add(OSAVIp2.multiply(38.7))
            .subtract(SR.multiply(0.004)).add(3.061))

        # Default model:
        lai = lai1.where(MSR.gt(1.1384), lai2)
        lai = lai.rename(band_name)

        if additional_models:
            # Compute two additional models and output a linear combination 
            # of the three models instead of the default model.
            # Requires the green band

            # Green normalized difference vegetation index (GNDVI):
            GNDVI = (nir.subtract(green)).divide(nir.add(green))
            GNDVIp2 = GNDVI.pow(2)
            # Green simple ratio (GSR):
            GSR = nir.divide(green)
            # Modified triangular vegetation index 2 (MTVI2):
            mtvi2root = ((nir.multiply(2).add(1)).pow(2)
                        .subtract(nir.multiply(6))
                        .add(nir.sqrt().multiply(5))
                        .subtract(0.5))
            MTVI2 = (((nir.subtract(green)).multiply(1.2)
                    .subtract((red.subtract(green)).multiply(2.5)))
                    .multiply(1.5).divide(mtvi2root.sqrt()))
            MTVI2p2 = MTVI2.pow(2)

            # rules
            additional_model1_lai1 = (
            (EVI2.multiply(3.92)).add(EVI2p2.multiply(21.57))
            .add(MSR.multiply(5.35))
            .add(NDVI.multiply(4.22)).add(NDVIp2.multiply(9.93))
            .subtract(NDWI.multiply(6.15)).subtract(NDWIp2.multiply(8.6))
            .subtract(NDWI2.multiply(0.53)).add(NDWI2p2.multiply(0.48))
            .subtract(OSAVI.multiply(7.01)).subtract(OSAVIp2.multiply(25.32))
            .subtract(SR.multiply(0.342)).subtract(2.9213822))

            additional_model1_lai2 = (
            (EVI2.multiply(94.29)).subtract(EVI2p2.multiply(31.69))
            .add(MSR.multiply(3.31))
            .add(NDVI.multiply(88.72)).subtract(NDVIp2.multiply(18.62))
            .subtract(NDWI.multiply(5.32)).subtract(NDWIp2.multiply(2.23))
            .subtract(NDWI2.multiply(20.64)).add(NDWI2p2.multiply(17.36))
            .subtract(OSAVI.multiply(173.11)).add(OSAVIp2.multiply(33.64))
            .subtract(SR.multiply(0.003)).add(3.5351541))

            additional_model1_lai = additional_model1_lai1.where(MSR.gt(1.1384), 
                additional_model1_lai2)

            additional_model2_lai1 = (
            (EVI2.multiply(5.52)).add(EVI2p2.multiply(29.04))
            .add(NDVI.multiply(4.73))
            .subtract(OSAVI.multiply(12.68)).subtract(OSAVIp2.multiply(28.13))
            .add(GNDVI.multiply(1.03)).add(GNDVIp2.multiply(2.97))
            .subtract(GSR.multiply(0.719))
            .subtract(MTVI2.multiply(0.31)).subtract(MTVI2p2.multiply(0.41))
            .add(SR.multiply(1.826)).subtract(1.1382301))

            additional_model2_lai2 = (
            (EVI2.multiply(66.55)).subtract(EVI2p2.multiply(8.07))
            .add(NDVI.multiply(44.51)).add(NDVIp2.multiply(13.99))
            .subtract(OSAVI.multiply(117.51)).subtract(OSAVIp2.multiply(7.91))
            .add(GNDVI.multiply(5.1)).subtract(GNDVIp2.multiply(12.33))
            .add(GSR.multiply(0.968))
            .add(MTVI2.multiply(8.01)).subtract(MTVI2p2.multiply(6.26))
            .subtract(SR.multiply(0.052)).add(0.028714))

            additional_model2_lai = additional_model2_lai1.where(GSR.gt(2.41101), 
                additional_model2_lai2)

            # linear combination of default and additional models
            lai = ((lai.multiply(0.4)).add(additional_model1_lai.multiply(0.3))
                  .add(additional_model2_lai.multiply(0.3)))

    else:
        # Compute vegetation indices used in this model
        # EVI, EVI2, MSR, NDVI, NDWI, NDWI2, OSAVI, SR
        # as defined in Houborg et al., 2018

        # Enhanced VI (EVI):
        EVI = np.minimum(2.5*(nir-red)/(nir+6*red-7.5*blue+1),0.99)
        # 2-band EVI:
        EVI2 = 2.5*(nir-red)/(nir+2.4*red+1) 
        # Mid-infrared simple ratio (MSR):
        MSR = nir/swir1
        # Normalized difference vegetation index (NDVI):
        NDVI = (nir-red)/(nir+red)
        # Normalized difference water index (NDWI):
        NDWI = (nir-swir1)/(nir+swir1) + 0.24 
        # Normalized difference water index 2 (NDWI2):
        NDWI2 = (nir-swir2)/(nir+swir2) + 0.22 
        # Optimized soil adjusted vegetation index (OSAVI):
        OSAVI = (nir-red)*1.16/(nir+red+0.16)
        # Simple ratio:
        SR = nir/red

        # rules:
        lai1 = 1.5*EVI-3.3*(EVI**2)+0.8*EVI2+31.56*(EVI2**2)+3.8*MSR\
            + 2.45*NDVI + 15.87*(NDVI**2) - 4.38*NDWI - 5.96*(NDWI**2)\
            - 0.38*NDWI2 + 0.27*(NDWI2**2) - 3.64*OSAVI - 36.35*(OSAVI**2)\
            - 0.456*SR - 1.852
        lai2 = -0.87*EVI-2.95*(EVI**2)+94.43*EVI2-29.5*(EVI2**2)+3.45*MSR\
            + 87.32*NDVI - 19.65*(NDVI**2) - 5.34*NDWI - 2.35*(NDWI**2)\
            - 18.08*NDWI2 + 14.54*(NDWI2**2) - 172.66*OSAVI + 38.7*(OSAVI**2)\
            - 0.004*SR + 3.061

        # model:
        lai = np.where(MSR<=1.1384, lai1, lai2)
        lai = np.maximum(lai, 0)

        if additional_models:
            # Compute two additional models and output a linear combination 
            # of the three models instead of the default model.
            # Requires the green band

            # Green normalized difference vegetation index (GNDVI):
            GNDVI = (nir-green)/(nir+green)
            # Green simple ratio (GSR):
            GSR = nir/green
            # Modified triangular vegetation index 2 (MTVI2):
            MTVI2 = 1.5*(1.2*(nir-green)-2.5*(red-green))\
                    /np.sqrt((2*nir+1)**2-(6*nir-5*np.sqrt(red))-0.5)
            MTVI2 = np.minimum(MTVI2, 0.99)
            # rules
            additional_model1_lai1 = 3.92*EVI2+21.57*(EVI2**2)+5.35*MSR\
                       + 4.22*NDVI + 9.93*(NDVI**2) - 6.15*NDWI - 8.6*(NDWI**2)\
                       - 0.53*NDWI2 + 0.48*(NDWI2**2) - 7.01*OSAVI - 25.32*(OSAVI**2)\
                       - 0.342*SR - 2.9213822

            additional_model1_lai2 = 94.29*EVI2-31.69*(EVI2**2)+3.31*MSR\
                       + 88.72*NDVI - 18.62*(NDVI**2) - 5.32*NDWI - 2.23*(NDWI**2)\
                       - 20.64*NDWI2 + 17.36*(NDWI2**2) - 173.11*OSAVI + 33.64*(OSAVI**2)\
                       - 0.003*SR + 3.5351541

            additional_model1_lai = np.where(MSR<=1.1384,  
                                    additional_model1_lai1,
                                    additional_model1_lai2)
            additional_model1_lai = np.maximum(additional_model1_lai, 0)


            additional_model2_lai1 = 5.52*EVI2 + 29.04*(EVI2**2) \
                       + 4.73*NDVI - 12.68*OSAVI - 28.13*(OSAVI**2)\
                       + 1.03*GNDVI + 2.97*(GNDVI**2) - 0.719*GSR\
                       - 0.31*MTVI2 - 0.41*(MTVI2**2) \
                       + 1.826*SR - 1.1382301

            additional_model2_lai2 = 66.55*EVI2 - 8.07*(EVI2**2) \
                       + 44.51*NDVI - 117.51*OSAVI - 7.91*(OSAVI**2)\
                       + 5.1*GNDVI - 12.33*(GNDVI**2) + 0.968*GSR\
                       + 8.01*MTVI2 - 6.26*(MTVI2**2) \
                       + 13.99*(NDVI**2) - 0.052*SR + 0.028714

            additional_model2_lai = np.where(GSR<=2.41101,  
                                    additional_model2_lai1,
                                    additional_model2_lai2)
            additional_model2_lai = np.maximum(additional_model2_lai, 0)
            # linear combination of default and additional models
            lai = 0.4*lai + 0.3*additional_model1_lai + 0.3*additional_model2_lai
    return lai


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
        if not hasattr(LAI, "size"):
            LAI = np.array(LAI)

        f_theta = (
            1 - np.exp(-0.5*LAI/np.cos(np.radians(theta))) 
        ).clip(0.05,0.9)

        if hasattr(f_theta, "rename"):
            f_theta = f_theta.rename("f_theta")

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
    high LAI (LAI >= LAI_thre): k=0.8
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
            if hasattr(LAI, "where"):
                k = (LAI.where(LAI<LAI_thre, 0.45)
                     .where(LAI>=LAI_thre,0.8))
            else:
                k = np.ones_like(LAI)
                k = np.where(LAI<LAI_thre, 0.8, 0.45)

        exp_term = -k*LAI
        if use_zenith:
            zenith, _ = solar_angles
            exp_term = exp_term/(np.sqrt(2*np.cos(zenith*DTORAD)))
        Rns = Rn*np.exp(exp_term)

        if hasattr(Rns, "rename"):
            Rns = Rns.rename("Rns")
    return Rns