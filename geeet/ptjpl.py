"""
This module contains functions to run the PT-JPL crop water use model.

The model functions are hybrid: they work both with numpy arrays or ee.Images*:
If the instance of the input is recognized as an ee.Image, the output is
computed as an ee.Image as well.
Otherwise, it is computed as a numpy array.

To reduce the dependencies of this package, 
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
from geeet.common import is_img
from geeet.meteo import compute_met_params, relative_humidity
from geeet.vegetation import compute_ftheta
try: 
    import ee
except Exception:
    pass


def ptjpl_arid(img=None, # ee.Image with inputs as bands (takes precedence over numpy arrays)
    Ta=None,  P=None, NDVI=None, F_aparmax=None,
    LAI = None, # optional; computed using NDVI if LAI is not available.
    Rn=None, Sdn=None, Ldn=None, Tr=None, Alb = None,  # net radiation OR Sdn, Ldn, Tr, and Albedo 
    G = None, # optional; computed if not provided. 
    RH=None, Td=None, # relative humidity OR dewpoint temperature (if RH not provided)
    doy=None, time=None, longitude=None,
    AlphaPT=1.26, # Default Priestly-Taylor coefficient for canopy potential transpiration
    G_params = [0.31, 74000, 10800], k_par = 0.5, k_rns = 0.6, Beta = 1.0, 
    Mask=1.0, Mask_Val=0.0, Vza = 0):
    '''
    Function to compute evapotranspiration components using the PT-JPL model
    adapted for arid lands (Aragon et al., 2018)

    Inputs ([] denotes an optional input that is computed if not provided.
            {} denotes an optional input that is required if the previous [] was not provided)
    img: ee.Image with the following bands (or image properties):
        - air_temperature : air temperature in Kelvin.
        - surface_pressure : Pressure in Pa.
        - NDVI : Normalized Difference Vegetation Index values.
        - [LAI]: Leaf area index (m2/m2) (computed using NDVI if not provided)
        - fapar_max : Maximum fapar* values.
        - [net_radiation]: Net radiation in W/m2 (Computed if not provided - see geeet.solar.compute_Rn)
        - {solar_radiation}: Downwelling shortwave radiation in W/m2 (Required if net_radiation is not found)
        - {thermal_radiation}: Downwelling longwave radiation in W/m2 (Required if net_radiation is not found)
        - {radiometric_temperature}: Surface temperature in K (Required if net_radiation is not found) 
        - {albedo}: shortwave broadband albedo (Required if net_radiation is not found)
        - [relative_humidity]: relative humidity (%) (computed if not provided - see geeet.meteo.relative_humidity)
        - {dewpoint_temperature}:  dewpoint temperature (K) (Required if relative_humidity is not provided)
        - [ground_heat_flux]: Ground heat flux, in W/m2 (optional; computed if not provided)
        - doy (ee.Image property): Day of year. 
        - time (ee.Image property): Local time of observation (decimal)    
    or: numpy arrays as keyword parameters (all of these are ignored if img is an ee.Image):
        - Ta: air temperature in Kelvin
        - P: surface pressure in Pa
        - NDVI: NDVI values
        - [LAI]: Leaf area index (m2/m2) (optional)
        - F_aparmax: Maximum fapar* values
        - [Rn]: net radiation in W/m2]. Alternatively:
            {Sdn}: Downwelling shortwave radiation in W/m2,
            {Ldn}: Downwelling longwave radiation in W/m2,
            {Tr}: radiometric temperature in K
            {Alb}: albedo
        - [RH]: relative humidity (%). Alternatively:
            {Td}: dewpoint temperature in K
        - doy: day of year
        - time: local time of observation (decimal)
        - longitude: longitude for each observation

        *Fraction of the photosynthetic active radiation (PAR) absorbed by green vegetation cover.

    Optional_Inputs: 
        - AlphaPT (float): Priestley-Taylor alpha coefficient
        - G_params (list [float A, float B, float C]): the parameters for
          computing solid heat flux (Santanello and Friedl, 2003)
          where A is the maximum ratio of G/Rns
          B reduces deviation of G/Rn to measured values, also thought of
          as the spread of the cosine wave and C is the phase shift between
          the peaks of Rns and G. B and C are in seconds.        
        - k_par (float): parameter used in the computation of LAI (see compute_lai) 
        - k_rns (float): parameter used to partition net radiation to soil 
                         (see geeet.vegetation.compute_rns)
        - Beta (float): Sensibility parameter to VPD in kPa (see compute_fsm)
        - Mask (numpy array): an array containing the coded invalid data (e.g. -9999)
            of the same size as the input image. Only for numpy array inputs; ignored for
            ee.Image inputs.
        - Mask_Val (number): the value that represents invalid data. Only for numpy array
            inputs; ignored for ee.Image inputs. 
        - band_names (list of strings): If provided, the bands in the output ee.Image
                are renamed using this list. Ignored if inputs are numpy arrays.            

    Outputs: 
        - ET (dictionary or ee.Image): ndictionary containing numpy arrays with the following
                                  components, or the following bands are added to the input image:
        -     LE: the latent heat flux.
        -     LEc: the canopy transpiration component of LE.
        -     LEs: the soil evaporation component of LE.
        -     LEi: the interception evaporation component of LE.
        -     H: the sensible heat flux.
        -     G: the ground heat flux.
        -     Rn: the net radiation.
    '''
    import numpy as np
    from geeet.vegetation import compute_Rns, compute_lai, compute_fwet, \
        compute_fg, compute_ft_arid, compute_fapar, compute_fm, \
        compute_fsm
    from geeet.solar import compute_g, compute_Rn
    from geeet.meteo import relative_humidity

    if is_img(img):
        band_names = img.bandNames()
        NDVI = img.select('NDVI')
        Ta = img.select('air_temperature')
        P = img.select('surface_pressure')
        F_aparmax = img.select('fapar_max')
        time = img.get('time')
        doy = img.get('doy')

        LAI = ee.Algorithms.If(band_names.contains('LAI'), img.select('LAI'), compute_lai(NDVI, k_par, band_name = 'LAI'))
        LAI = ee.Image(LAI)

        RH = ee.Algorithms.If(band_names.contains('relative_humidity'), img.select('relative_humidity'),\
            relative_humidity(Ta, img.select('dewpoint_temperature'), P))
        RH = ee.Image(RH)

        f_theta = compute_ftheta(LAI, theta=Vza)       

        Rn = ee.Algorithms.If(band_names.contains('net_radiation'), img.select('net_radiation'),\
            compute_Rn(img.select('solar_radiation'), img.select('thermal_radiation'), img.select('albedo'),\
                img.select('radiometric_temperature'), f_theta))
        Rn = ee.Image(Rn)

        rns = compute_Rns(Rn, LAI, k=k_rns, use_zenith = False)

        G = ee.Algorithms.If(band_names.contains('ground_heat_flux'), img.select('ground_heat_flux'), \
            compute_g(doy = doy, time=time, Rns = rns, G_params = G_params))
        G = ee.Image(G)

    # all of these functions work for both ee.Images and numpy arrays:
    fwet = compute_fwet(RH) 
    fg = compute_fg(NDVI)
    ft = compute_ft_arid(Ta)
    f_apar = compute_fapar(NDVI)
    fm = compute_fm(f_apar, F_aparmax)
    met_params = compute_met_params(Ta, P)  
    fsm = compute_fsm(RH, Ta, Beta)

    if is_img(img):
        taylor = met_params.select('taylor').multiply(AlphaPT)
        rnc = Rn.subtract(rns)
        cst_1 = ee.Image(1.0)
        fwet_sub1 = cst_1.subtract(fwet)
        # Canopy transpiration image
        LEc = fwet_sub1.multiply(fg).multiply(ft).multiply(fm).multiply(taylor).multiply(rnc).rename('LEc')
        # Soil evaporation image
        LEs = ((fwet_sub1.multiply(fsm)).add(fwet)).multiply(taylor).multiply(rns.subtract(G)).rename('LEs')
        # Interception evaporation image
        LEi = fwet.multiply(taylor).multiply(rnc).rename('LEi')
        # Evapotranspiration image
        LE = LEc.add(LEs).add(LEi).rename('LE')
        # Compute the sensible heat flux image H by residual
        H = (Rn.subtract(G)).subtract(LE).rename('H')
        Rn = rns.add(rnc).rename('Rn')

        # Add the outputs to the input image:
        # LE, LEc, LEs, LEi, H, G 
        G = G.rename('G')
        ET = img.addBands(LE).addBands(LEc).addBands(LEs).addBands(LEi).addBands(H).addBands(G).addBands(Rn)
    else:
        _,_,_,_,_,_,_,taylor = met_params  # q,ea,rho,cp,delta,Lambda,gamma,taylor
        taylor = taylor*AlphaPT

        if LAI is None:
            LAI = compute_lai(NDVI, k_par)

        if RH is None:
            RH = relative_humidity(Ta, Td, P)

        if Rn is None:
            f_theta = compute_ftheta(LAI, Vza)
            Rn = compute_Rn(Sdn, Ldn, Alb, Tr, f_theta)

        rns = compute_Rns(Rn, LAI, k=k_rns, use_zenith = False)
        if G is None:
            G = compute_g(doy = doy, time=time, Rns = rns, G_params = G_params, longitude=longitude) 
        
        rnc = Rn-rns
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
         
        ET=LE, LEc, LEs, LEi, H, G, Rn
        et_keys = ['LE', 'LEc', 'LEs', 'LEi', 'H', 'G', 'Rn']
        ET = dict(zip(et_keys, ET))
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