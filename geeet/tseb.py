"""
This module contains functions to run the TSEB crop water use model.

The model functions are hybrid: they work both with numpy arrays or ee.Images:
If the instance of the input is recognized as an ee.Image, the output is
computed as an ee.Image as well.
Otherwise, it is computed as a numpy array.

To reduce the dependencies of this package, 
the ee capabilities are optional. That means that 
in order to use the ee capabilities, the user must install the ee package:

conda install -c conda-forge earthengine-api
or 
pip install earthengine-api

References can be found at the end of this module
They can be printed in python using the following two functions:
geeet.tseb.cite() - main reference for this module
geeet.tseb.cite_all() - all references used for this module
"""

import numpy as np
from geeet.common import is_img
try: 
    import ee
except Exception:
    pass


def init_canopy(LAI, ch=0.3, min_LAI = 1.0, D_0_min = 0.004, fd=0.65):
    """
    Initialize canopy height array (or ee.Image)
    If ch is a scalar, this function returns ch as
    either an ee.Image (if LAI is ee.Image)
    or as a numpy array (if LAI is numpy array)
    the values for ch will be constant (defaults to 0.3)
    for high LAI values (>min_LAI),
    while for low LAI values (<=min_LAI), ch will be calculated
    using a minimum value for the zero displacement height (m):
    D_0_min*3/2.0 (defaults to 0.006 because D_0_min defaults to 0.004)

    Note: this gives only two discrete values.. 
    """
    if is_img(LAI):
        if not is_img(ch):
            laiMask = LAI.mask()
            ch = ee.Image(ch)
            ch = ch.updateMask(laiMask)
            ch = ch.where(LAI.lte(min_LAI), D_0_min/fd)
    else:
        if not isinstance(ch, np.ndarray):
            ch = np.array(np.ones_like(LAI)*ch)
            ch[LAI <= min_LAI] = D_0_min/fd

    return ch

def tseb_series(img=None,    # ee.Image with inputs as bands (takes precedence over numpy arrays)
                Tr=None, NDVI=None, LAI = None,  # numpy arrays
                P = None, Ta = None, U = None, Sdn = None, Ldn = None, Rn = None,
                Alb = None,
                doy = None, time = None, Vza = None, longitude = None, latitude=None,
                CH = 0.3, F_g = 1.0, k_par = 0.5, k_rns=0.45,    
                AlphaPT = 1.26, # Default Priestly-Taylor coefficient for canopy potential transpiration
                G_params = [0.31, 74000, 10800],
                Leaf_width = 0.1, zU = None, zT = None,
                max_iterations = 5
                ):
    """
    Priestley-Taylor Two Source Energy Balance Model
    Function to compute TSEB energy fluxes using a single observation
    of composite radiometric temperature using resistances in series (i.e. for dense vegetation). 

    All inputs are given as keyword parameters. 

    Inputs:
    Either a single ee.Image (img) with the following inputs as bands (unless otherwise specified) 
    or inputs as numpy arrays (unless otherwise specified):

    - Radiometric temperature (Kelvin) e.g. Land surface temperature:   radiometric_temperature (band) or Tr (numpy array)
    - Normalized Difference Vegetation Index (NDVI): NDVI (band) or NDVI (numpy array)
    - Surface pressure (Pa): surface_pressure (band) or P (numpy array)
    - Air temperature (K): air_temperature (band) or Ta
    - Wind speed (m/s): wind_speed (band) or U
    - Downwelling shortwave radiation (W/m2): solar_radiation (band) or Sdn (numpy array)
    - Downwelling longwave radiation (W/m2): thermal_radiation (band) or Ldn (numpy array) 

    Alternative inputs (TODO)
    These inputs could be supplied instead of computed - NOT YET IMPLEMENTED
    - Net radiation (W/m2): net_radiation (band) or Rn (numpy array) TODO  - could replace Sdn, Ldn, albedo
    - Leaf Area Index (LAI; m2/m2): LAI (band) or LAI (numpy array)  TODO - include check if LAI is given 
            (currently it is computed by default)

    Scalar inputs:
    - zU: Height of wind speed measurement (m) (or height of the "surface" level if data is from a climate model)
    - zT: Height of measurement of air temperature (m) (or height of the "surface" level if data is from a climate model)
    - CH: canopy height in meters
    - Leaf_width: Parameter small 's' in Norman et al., 1995 - leaf size in meters (e.g. Equation A.8)
    - AlphaPT: Priestly Taylor coefficient for canopy potential transpiration (defaults to 1.26)
    - F_g: fraction of vegetation that is green: default to 1.0
    - doy: day of year of observation - either a scalar or a property in img
    - time: time of observation - either a scalar or a property in img
    - Vza: viewing zenith angle - either a scalar or a property in img
    - max_iterations: Maximum number of iterations allowed for the iterative process.
    - k_par: parameter controlling the calculation of LAI from NDVI
    - k_rns: parameter controlling the partitioning of net radiation to the soil/vegetation.
    - G_params: list of length 3 containing the parameters for the calculation of soil heat flux:
                 [maximum ratio of G/Rns,  spread of the cosine wave,  phase shift ]
                 (Santanello and Friedl, 2003)

    Outputs:
    The following bands are added to the input image (img) or 
    a dictionary with the following numpy arrays is returned:
    - LE: Total latent heat flux, in W/m2
    - LEs: latent heat flux from the soil component, in W/m2
    - LEc: latent heat flux from the canopy component, in W/m2
    - H: Total sensible heat flux, in W/m2
    - Hs: sensible heat flux from the soil component, in W/m2
    - Hc: sensible heat flux from the canopy component, in W/m2
    - G: ground heat flux, in W/m2
    - Rn: net radiation, in W/m2
    - Rns: net radiation partitioned to the soil surface, in W/m2
    - Rnc: net radiation partitioned to the canopy, in W/m2
    - Ts: soil surface temperature, in K
    - Tc: canopy temperature, in K
    - Tac: temperature of air in the canopy layer, in K
    - Ra: aerodynamic resistance, in m s-1
    - Rs: resistance to transport of heat between soil surface and a height
          representing the canopy, in m s-1
    - Rx: total boundary layer resistance of the complete canopy of leaves, 
          in m s-1
    - iteration: number of iterations to reach a positive LEs
    """
    from geeet.vegetation import compute_lai, compute_ftheta, compute_Rns
    from geeet.solar import compute_solar_angles, compute_Rn, compute_g
    from geeet.meteo import compute_met_params, compute_roughness
    from geeet.resistances import RN95
    from geeet.MOST import Ustar as compute_ustar
    from geeet.MOST import MOL

    # Hard-coded scalar parameters
    min_LAI = 1.0 # Threshold to use minimum D0 and ZM values:
    D_0_min = 0.004 # Minimum zero-plane displacement height (m).
    z_0M_min = D_0_min/0.65 * 0.125#0.003  # Minimum value for the aerodynamic surface roughness length (m).

    if is_img(img):
        from ee import Number
        # Required
        Tr = img.select('radiometric_temperature') # in K
        P = img.select('surface_pressure')   # in Pa
        Ta = img.select('air_temperature')   # in K
        U = img.select('wind_speed')         # in m/s
        Alb = img.select('albedo')  
        Sdn = img.select('solar_radiation')  # in W/m2
        Ldn = img.select('thermal_radiation') # in W/m2
        NDVI = img.select('NDVI')
        time = Number(img.get('time'))
        doy = Number(img.get('doy'))
        Vza = Number(img.get('viewing_zenith')) # Viewing Zenith angle (0 for Landsat)
        # Optional
        #LAI = img.select('LAI')  # TODO: check if band exists
        #CH = img.select('canopy_height') # TODO: check if band exists
        #Rn = img.select('net_radiation') # TODO: check if exists, then skip calculation of Rn from Sdn, Ldn
        
    # The following functions are designed to work
    # for both numpy and ee.Image inputs:
    LAI = compute_lai(NDVI, k_par) # TODO: check if LAI is given as an input and skip this compute_lai.
                                   # TODO: add option to compute lai using Houborg et al. trained models (Cubist and RF)
    f_theta = compute_ftheta(LAI, theta=Vza) # Fraction of field of view of sensor occupied by canopy (e.g. ~0 for LAI=0; ~0.4 for LAI=1, ~1 for LAI>6 with theta=0)
    Rn = compute_Rn(Sdn, Ldn, Alb, Tr, f_theta) # Net radiation with the upwards thermal radiation based on surface temperature (proportional to Tr**4)
                                                  # TODO: check if Rn is provided and skip this compute_Rn.
    CH = init_canopy(LAI, CH, min_LAI, D_0_min) # Initializes canopy height based on LAI  
    rough = compute_roughness(CH, min_values = [z_0M_min, z_0M_min, D_0_min])  # Computes aerodynamic roughness length for momentum and heat tranposrt (Z_0M, Z_0H) and
                                                # zero-plane displacement height. 
                                       # rough contains the results either as bands ('ZM', 'ZH', 'D0' if rough is ee.Image)
                                       # or as a list: [Z0M, Z0H, D0]
    solar_angles = compute_solar_angles(doy, time, longitude, latitude) # Sun zenith and azimuth for the specified lon/lat/date/time
                                                  # Both parameters returned either as bands (if img is not None)
                                                  # or as a list: [zenith, azimuth]
                                                  # note: if ee.Image, longitude and latitude are None and ignored (not needed)
    Rns = compute_Rns(Rn, LAI, solar_angles, use_zenith=True, k=k_rns) # Soil net radiation Rns = E*Rn where E is either just exp(-kLAI)
                                                                      #                                or exp(-kLAI/sqrt(2cos(Zenith))))
                                                                      # here in TSEB we use the second parameterization (use_zenith = True)
    G = compute_g(doy = doy, time=time, Rns = Rns, G_params = G_params, longitude=longitude) # Soil heat flux (Santanello et al., 2003)
    met_params = compute_met_params(Ta, P) # computes the following meteorological parameters:
                                             # q: specific humidity (dimensionless)
                                             # ea: water vapor pressure, in Pa
                                             # rho: air density, in kg m-3
                                             # cp: air heat capacity, in J kg-1 K-1
                                             # s: slope of water vapor pressure v T curve, in Pa K-1
                                             # L: Latent heat of vaporization, in MJ kg -1
                                             # psicr: Psicrometric constant, in Pa K-1


    # Initial calculations assuming stable conditions ( L -> ∞ )
    ustar = compute_ustar(U, zU, L=1e10, rough_params = rough)
    resist = RN95(U, CH, rough, LAI, Leaf_width, zU, zT, Ustar = ustar, L=1e10) # computes three resistances: Ra, Rs, and Rx (see geeet.resistances.RN95)


    if is_img(img):
        from ee import Image, List, ImageCollection
        # Parameters from hybrid functions:
        taylor = met_params.select('taylor')  # delta / (delta + gamma)
        rho = met_params.select('rho')
        Lambda = met_params.select('Lambda')
        cp = met_params.select('cp')
        ra = resist.select('Ra')
        rs = resist.select('Rs')
        rx = resist.select('Rx')

        # Canopy net radiation
        Rnc = Rn.subtract(Rns)

        # Initialize temperature partitioning (soil and canopy)
        # using the PT-equation:
        LEc = taylor.multiply(Rnc).multiply(AlphaPT*F_g) # Canopy latent heat flux (N95 equation 12)
        LEc = Rnc.multiply(AlphaPT*F_g).multiply(taylor)
        LEc = LEc.where(LEc.lt(0), Image(0))
        Hc = Rnc.subtract(LEc)  # Canopy sensible heat flux (N95 equation 14)
        Tc = (ra.multiply(Hc).divide(rho.multiply(cp))).add(Ta) # in K (N95 equation 14)
        Ts = (Tr.subtract(f_theta.multiply(Tc))).divide(Image(1).subtract(f_theta)) #(N95 A.5)

        # Constraints on Tc and Ts based on DisALEXI
        Ts = Ts.where(f_theta.gte(0.9), Tr)
        Ts = Ts.where(f_theta.lte(0.1), Tr)
        Tc = Tc.where(f_theta.gte(0.9), Tr)
        Tc = Tc.where(f_theta.lte(0.1), Tr)

        # Initial fluxes using TSEB-parallel
        Hs = rho.multiply(cp).multiply(Ts.subtract(Ta)).divide(ra.add(rs))
        LEs  = (Rns.subtract(G)).subtract(Hs)

        # initialize Tac = Tc
        Tac = Tc 
        LE = LEc.add(LEs)
        H = Hc.add(Hs)

        # Prepare an initial image containing all the variables that need to be updated iteratively:
        initialImg = Tc.addBands(Ts).addBands(Tac).addBands(Hc).addBands(Hs).addBands(LEc).addBands(LEs)\
        .addBands(ra).addBands(rs).addBands(rx).addBands(ustar).addBands(Image(AlphaPT)).addBands(Image(0))
        initialImg = initialImg.rename('Tc','Ts','Tac','Hc','Hs','LEc','LEs','Ra','Rs','Rx',\
            'Ustar','alphaPT','iteration')
        iterStart = ee.List([initialImg]) # Initial list only contains the initialImg 
        
        # Prepare a dummy image collection for the iterative procedure:
        numList = List.repeat(1, max_iterations)
        def zeroImg(f):
            return Image(0)
        dummyIC = ImageCollection(numList.map(zeroImg))

        ##### Iterative procedure
        def tseb_series_iteration(img, list):
            '''
            Perform one iteration of the TSEB series algorithm to
            update the temperatures (Tc, Ts, Tac), heat fluxes (H, LE), and resistances
            The updated values are added to a new image, and this new image is
            appended at the end of "list". 
            The updated values are kept only for pixels where the previous 
            iteration LEs (soil latent heat flux) is negative. 

            Specifically, it computes equations in the Norman et al., 1995 model (N95):
            Equation 1, 12, A.5 - A.13

            Inputs:
                - img: Not needed/not used.  
                - list: list of ee.Images. Each ee.Image contains the following bands:
                        - Tc: Temperature of the canopy (K)
                        - Ts: Temperature of the soil surface (K)
                        - Tac: Temperature of the air in the canopy layer (K)
                        - Hc: Sensible heat flux from the canopy source (W/m2)
                        - Hs: Sensible heat flux from the soil source (W/m2)
                        - H: Total sensible heat flux (W/m2)
                        - LEc: Latent heat flux from the canopy source (W/m2)
                        - LEs: Latent heat flux from the soil source (W/m2) 
                        - Ra: aerodynamic resistance, in m s-1
                        - Rs: resistance to transport of heat between soil surface and a height
                              representing the canopy, in m s-1
                        - Rx: total boundary layer resistance of the complete canopy of leaves, 
                              in m s-1
                        - Ustar: friction velocity (U*)
                        - L: Monin-Obukhov length 
                        - alphaPT: Priestly-Taylor coefficient
            '''
            ### Previous iteration values (subscript 'o' for old):
            oldImg = ee.Image(ee.List(list).get(-1))
            # Here I only select the values that are needed explicitly:
            # since the .where will do it.. i.e. result = oldImg.where(oldImg.select('LEs').lt(0), updatedImg) 
            iterationo = oldImg.select('iteration')   
            Hco = oldImg.select('Hc')
            LEso = oldImg.select('LEs')
            rao = oldImg.select('Ra')
            rso = oldImg.select('Rs')
            rxo = oldImg.select('Rx')
            ustaro = oldImg.select('Ustar') 
            AlphaPTo = oldImg.select('alphaPT')
        
            ##### Update values (subscript 'u' for updated): 
            iterationu = iterationo.add(1)
            AlphaPTu = AlphaPTo.subtract(0.01)
            AlphaPTu = AlphaPTu.where(AlphaPTu.lt(0), 0)
        
            # Temperatures (Tc, Ts, Tac) using N95 equations (1, A.7-A.13)
            # Linear approximation of Tc (N95 A7):
            # Note that Hc = Rnc - LEc   where LEc = Rnc[alphaPT*Fg*taylor], so 
            # Hc = Rnc * (1 - alphaPT * Fg * taylor)
            Tclin_num1 = Ta.divide(rao)
            Tclin_num2 = Tr.divide(rso.multiply(Image(1).subtract(f_theta)))
            Tclin_num3 = (Hco.multiply(rxo).divide(rho.multiply(cp))).multiply((Image(1).divide(rao)).add(Image(1).divide(rso)).add(Image(1).divide(rxo)))
            Tclin_denom = (rao.pow(-1)).add(rso.pow(-1)).add(f_theta.divide(rso.multiply(Image(1).subtract(f_theta))))
            Tclin = (Tclin_num1.add(Tclin_num2).add(Tclin_num3)).divide(Tclin_denom)
            # N95 equation A.12 (TD):
            Td1 = Tclin.multiply(Image(1).add(rso.divide(rao)))
            Td2 = (Hco.multiply(rxo).divide(rho.multiply(cp))).multiply(Image(1).add(rso.divide(rxo)).add(rso.divide(rao)))
            Td3 = Ta.multiply(rso.divide(rao))
            Td = Td1.add(Td2).subtract(Td3)
            # N95 equation A.11 (deltaTc), i.e. correction to the linear approximation of Tc:
            dTc_num = (Tr.pow(4)).subtract(f_theta.multiply(Tclin.pow(4))).subtract((Image(1).subtract(f_theta)).multiply(Td.pow(4)))
            dTc_denom1 = (Td.pow(3)).multiply(rso.divide(rao).add(1)).multiply(4).multiply(Image(1).subtract(f_theta))
            dTc_denom2 = f_theta.multiply(4).multiply(Tclin.pow(3))
            dTc = dTc_num.divide(dTc_denom1.add(dTc_denom2))
            # N95 equation A.13 (Tc = Tclin + delta Tc)
            Tcu = Tclin.add(dTc).rename('Tc')
            # N95 equation 1, solving for (1-f(theta))*Ts^n  = Tr^n - f(theta)*Tc^n
            # the RHS of this equation is:
            TsRHS = Tr.pow(4).subtract(f_theta.multiply(Tcu.pow(4)))
            # Force TsRHS to be a positive value to avoid complex numbers:
            TsRHS = TsRHS.max(1e-6)
            # Estimate Ts (N95 equation 1)
            Tsu = (TsRHS.divide(Image(1).subtract(f_theta))).pow(0.25)
            Tsu = Tsu.max(1e-6)   # Force a minimum value for Ts 
            Tsu = Tsu.rename('Ts')
        
            # Estimate Tac (N95 equation A.4):
            Tacu = ((Ta.divide(rao)).add(Tsu.divide(rso)).add(Tcu.divide(rxo)))\
                    .divide((Image(1).divide(rao)).add(Image(1).divide(rso)).add(Image(1).divide(rxo)))
            Tacu = Tacu.rename('Tac')
        
            # Constraints on Tc and Ts based on DisALEXI
            Tsu = Tsu.where(f_theta.gte(0.9), Tr)
            Tsu = Tsu.where(f_theta.lte(0.1), Tr)
            Tcu = Tcu.where(f_theta.gte(0.9), Tr)
            Tcu = Tcu.where(f_theta.lte(0.1), Tr)
        
            ### Update sensible heat fluxes using the in-series network:
            Hcu = rho.multiply(cp).multiply(Tcu.subtract(Tacu)).divide(rxo).rename('Hc')   # Tc: new; Tac: new; rx: old 
            Hsu = rho.multiply(cp).multiply(Tsu.subtract(Tacu)).divide(rso).rename('Hs')   # Tc: new; Tac: new; rs: old
        
            ### Update latent heat fluxes as a residual of the energy balance
            LEsu = Rns.subtract(G).subtract(Hsu).rename('LEs') # Hs: new
            LEcu = Rnc.subtract(Hcu).rename('LEc')             # Hc: new
        
            ### Update total fluxes
            LEu = LEcu.add(LEsu).rename('LE')                # LEc: new; LEs: new
            Hu = Hcu.add(Hsu).rename('H')                   # Hc: new;  Hs: new
            
            ### Update M-O length (L) and friction velocity
            Lu = MOL(ustaro, Ta, rho, cp, Lambda, Hu, LEu)         #Ustar: old; H: new; LE: new 
            Lu = Lu.rename('L')
            ustaru = compute_ustar(U, zU, Lu, rough_params = rough)  # L: new; 
        
            ### Update resistances (rau, rsu, rxu)
            resistU = RN95(U, CH, rough, LAI, Leaf_width, zU, zT, Ustar = ustaru, L=Lu)  # Ustar: new; L: new
            rau = resistU.select('Ra')
            rsu = resistU.select('Rs')
            rxu = resistU.select('Rx')
        
            # Finally, recompute LEc using the PT equation (N95 equation 12)
            # for any pixel that still has LEs<0:
            LEcu2 = taylor.multiply(Rnc).multiply(AlphaPTu).multiply(F_g) 
            LEcu = LEcu.where(LEsu.lt(0), LEcu2)
            # and Hc = Rnc-LEc
            Hcu2 = Rnc.subtract(LEcu)
            Hcu = Hcu.where(LEsu.lt(0), Hcu2)
        
            # End of updates in this iteration
            # These are now included in "updatedImg", an image
            # containing all the updated values. However, note that we will
            # not necessarilly use the updated values everywhere - 
            # only where LEs<0 in the previous iteration.  
            updatedImg = Tcu.addBands(Tsu).addBands(Tacu)\
                .addBands(Hcu).addBands(Hsu).addBands(LEcu).addBands(LEsu)\
                .addBands(rau).addBands(rsu).addBands(rxu)\
                .addBands(ustaru).addBands(AlphaPTu).addBands(iterationu)
            updatedImg = updatedImg.rename('Tc','Ts','Tac','Hc','Hs',\
                'LEc','LEs','Ra','Rs','Rx','Ustar','alphaPT','iteration')
        
            # Finally, we select the values we will keep
            # (we only update pixels where LEs was previously negative)
            # These values will be the "old" Img in the next iteration
            resImg = oldImg.where(LEso.lt(0), updatedImg)
            return ee.List(list).add(resImg)
        ##### End of iterative procedure definition.
         
        # Call the iterative function using the dummy image collection and the initial estimates:
        iterListRes = ee.List(dummyIC.iterate(tseb_series_iteration, iterStart))
        resultImage = ee.Image(iterListRes.get(-1))  # The result is the last value of this list. 

        # Compute the total fluxes 
        LEs = resultImage.select('LEs')
        LEc = resultImage.select('LEc')
        LE = LEs.add(LEc).rename('LE')
        Hs = resultImage.select('Hs')
        Hc = resultImage.select('Hc')
        H = Hs.add(Hc).rename('H')
        resultImage = resultImage.addBands(LE).addBands(H)

        # Also return other fluxes:
        resultImage = resultImage.addBands(G.rename('G')).addBands(Rn.rename('Rn'))\
            .addBands(Rns.rename('Rns')).addBands(Rnc.rename('Rnc'))

        return img.addBands(resultImage)
    else:
        # Retrieve parameters from hybrid functions:
        _, _, rho, cp, _, Lambda, _, taylor = met_params    #[q, ea, rho, cp, s, Lambda, psicr, taylor]
        ra, rs, rx = resist

        # Canopy net radiation:
        Rnc = Rn - Rns

        # Initialize temperature partitioning (soil and canopy)
        # using the PT-equation:
        LEc = AlphaPT*F_g*taylor*Rnc # Canopy latent heat flux (N95 equation 12)
        LEc = np.array(LEc)

        LEc[LEc<0]=0.0
        Hc = Rnc - LEc # Canopy sensible heat flux (N95 equation 14)
        Tc = ra*Hc/(rho*cp) + Ta  # canopy T, in K (N95 equation 14)
        Ts = (Tr - f_theta*Tc)/(1-f_theta)# soil T, in K (N95 equation A.5, i.e. linearized equation 1)
        Tc = np.array(Tc)
        Ts = np.array(Ts)
        Tr = np.array(Tr)

        # Constraints on Tc and Ts based on DisALEXI
        Ts[f_theta >= 0.9] = Tr[f_theta >= 0.9]
        Ts[f_theta <= 0.1] = Tr[f_theta <= 0.1]
        Tc[f_theta >= 0.9] = Tr[f_theta >= 0.9]
        Tc[f_theta <= 0.1] = Tr[f_theta <= 0.1]

        # Initial fluxes estimation using TSEB in parallel
        Hs = rho*cp*(Ts-Ta)/(ra+rs)
        LEs = Rns - G - Hs
        LEs = np.array(LEs)

        Tac = Tc.copy()  # initialize Tac for the in-series network
        LE = LEc + LEs 
        H = Hc + Hs

        LEs = np.array(LEs)
        LEc = np.array(LEc)
        LE = np.array(LE)
        Hs = np.array(Hs)
        Hc = np.array(Hc)
        H = np.array(H)
        # Iterative procedure:
        # in each iteration, we only update pixels
        # where LEs was previously negative. 
        it = np.zeros_like(LE)
        for iteration in range(max_iterations):
            pixelsToUpdate = LEs<0 # in previous iteration, or from initialization
            it[pixelsToUpdate] = iteration+1
            if np.all(pixelsToUpdate == False):
                break

            #### Update stage:
            ### Update temperatures (Tc, Ts, Tac) using N95 equations (1, A.5-A.13)
            # Linear approximation of Tc (N95 A7):
            # Note that Hc = Rnc - LEc   where LEc = Rnc[alphaPT*Fg*taylor], so 
            # Hc = Rnc * (1 - alphaPT * Fg * taylor)
            Tclin = (Ta/ra + Tr/(rs*(1-f_theta)) + (1/ra+1/rs+1/rx)*Hc*rx/(rho*cp))\
                    /(1/ra+1/rs+f_theta/(rs*(1-f_theta)))

            # N95 equation A.12 (TD):
            Td = Tclin*(1+rs/ra) - (1+rs/rx+rs/ra)*Hc*rx/(rho*cp) - Ta*rs/ra
            # N95 equation A.11 (deltaTc), i.e. correction to the linear approximation of Tc:
            dTc = ((Tr**4) - f_theta*Tclin**4 - (1-f_theta)*Td**4)\
                / (4*(1-f_theta)*(Td**3)*(1+rs/ra)+4*f_theta*Tc**3)
            # N95 equation A.13 (Tc = Tclin + delta Tc)
            Tcu  = Tclin + dTc
            # N95 equation 1, solving for (1-f(theta))*Ts^n  = Tr^n - f(theta)*Tc^n
            # the RHS of this equation is:
            TsRHS = Tr**4 - f_theta*Tcu**4 
            # Force TsRHS to be a positive value to avoid complex numbers:
            TsRHS = np.maximum(TsRHS, 1e-6)
            # Estimate Ts (N95 equation 1)
            Tsu = (TsRHS/(1-f_theta))**0.25
            Tsu = np.maximum(Tsu, 1e-6) # Force a minimum value for Ts 

            # Estimate Tac (N95 equation A.4):
            Tacu = (Ta/ra + Tsu/rs + Tcu/rx)\
                /  (1/ra + 1/rs + 1/rx)

            # Constraints on Tc and Ts based on DisALEXI
            Tsu = np.array(Tsu)
            Tcu = np.array(Tcu)
            Tsu[f_theta>0.9] = Tr[f_theta>0.9]
            Tsu[f_theta<0.1] = Tr[f_theta<0.1]
            Tcu[f_theta>0.9] = Tr[f_theta>0.9]
            Tcu[f_theta<0.1] = Tr[f_theta<0.1]
           
            Tc[pixelsToUpdate] = Tcu[pixelsToUpdate]
            Ts[pixelsToUpdate] = Tsu[pixelsToUpdate]
            Tac[pixelsToUpdate] = Tacu[pixelsToUpdate]

            ### Update sensible and latent heat fluxes, now using the 
            # in-series network:
            Hcu = rho*cp*(Tc-Tac)/rx
            Hsu = rho*cp*(Ts-Tac)/rs
            LEsu = Rns - G - Hsu
            LEcu = Rnc - Hcu
            LEu = LEcu + LEsu
            Hu = Hcu + Hsu
            ### Update M-O length (L) and friction velocity
            L = MOL(ustar, Ta, rho, cp, Lambda, Hu, LEu) 
            ustaru = compute_ustar(U, zU, L, rough)
            ustar[pixelsToUpdate] = ustaru[pixelsToUpdate]

            ### Update resistances
            resistu = RN95(U, CH, rough, LAI, Leaf_width, zU, zT, Ustar=ustar, L=L)
            rau, rsu, rxu = resistu
            ra[pixelsToUpdate] = rau[pixelsToUpdate]
            rs[pixelsToUpdate] = rsu[pixelsToUpdate]
            rx[pixelsToUpdate] = rxu[pixelsToUpdate]
            ### Update the alpha PT constant:
            AlphaPT-=0.01
            AlphaPT = np.maximum(AlphaPT, 0) 

            # Finally, recompute LEc using the PT equation (N95 equation 12)
            # for any pixel that still has LEs<0:
            LEcu2 = taylor*Rnc*AlphaPT*F_g          
            nLEs = LEsu<0  # negative LEs
            LEcu = np.array(LEcu)
            LEcu2 = np.array(LEcu2)
            LEcu[nLEs] = LEcu2[nLEs]
            # and Hc = Rnc-LEc
            Hcu2 = Rnc-LEcu
            Hcu = np.array(Hcu)
            Hcu2 = np.array(Hcu2)
            Hcu[nLEs] = Hcu2[nLEs]

            # Update fluxes (H, L): 
            LEs[pixelsToUpdate] = LEsu[pixelsToUpdate]
            LEc[pixelsToUpdate] = LEcu[pixelsToUpdate]
            LE[pixelsToUpdate] = LEu[pixelsToUpdate]
            Hs[pixelsToUpdate] = Hsu[pixelsToUpdate]
            Hc[pixelsToUpdate] = Hcu[pixelsToUpdate]
            H[pixelsToUpdate] = Hu[pixelsToUpdate]

        et_tseb_out = LE, LEs, LEc, Hs, Hc, G, Rn, Rns, Rnc, Ts, Tc, Tac, ra, rs, rx, it
        et_tseb_out_keys = ['LE', 'LEs', 'LEc', 'Hs', 'Hc', 'G', 'Rn', 'Rns', 'Rnc', 'Ts', 'Tc', 'Tac', 'Ra', 'Rs', 'Rx', 'iteration']
        et_tseb_out = dict(zip(et_tseb_out_keys, et_tseb_out))
        return et_tseb_out

main_ref="Norman, J.M., et al. (1995). \"Source approach for estimating soil and \
vegetation energy fluxes in observations of directional radiometric surface \
temperature\". Agricultural and Forest Meteorology, 77(3), pp. 263-293. \
https://doi.org/10.1016/0168-1923(95)02265-Y"

# Citation:
# cite() - main reference only
def cite():
    print(main_ref)
#cite_all() - all references
def cite_all():
    for ref in all_refs:
        print(ref)

all_refs=["Campbell, G. S., & Norman, J. M. \
\"Introduction to environmental biophysics (2nd ed.) (1998)\"\
New York: Springer, pp. 168-169\
http://dx.doi.org/10.1007/978-1-4612-1626-1",
"Colaizzi, P. D., Kustas, William P., Anderson, Martha C., \
Agam, Nurit, Tolk, Judy A., Evett, Steven R., Howell, Terry A., \
Gowda, Prasanna H., and O\'Shaughnessy, Susan A. \
\"Two-source energy balance model estimates of evapotranspiration \
using component and composite surface temperatures\" (2012) \
Advances in Water Resources, 50, pp. 134-151. \
https://doi.org/10.1016/j.advwatres.2012.06.004",
"Kustas, W. P. and J. M. Norman  \
\"Evaluation of soil and vegetation heat flux predictions using a \
simple two-source model with radiometric temperatures for partial \
canopy cover\" (1999) Agricultural and Forest Meteorology, 94(1), \
pp. 13-29. https://doi.org/10.1016/S0168-1923(99)00005-2",
"Norman, J. M.,  Kustas, W. P., and Humes, K. S. \
\"Source approach for estimating soil and vegetation energy fluxes in \
observations of directional radiometric surface temperature\" (1995) \
Agricultural and Forest Meteorology, 77(3), pp. 263-293. \
https://doi.org/10.1016/0168-1923(95)02265-Y",
"Joseph A. Santanello Jr. and Mark A. Friedl. \
\"Diurnal Covariation in Soil Heat Flux and Net Radiation\" (2003) \
J. Appl. Meteor., 42, pp. 851-862. Remote Sensing of Environment, \
112 (3), pp. 901-919., \
http://dx.doi.org/10.1175/1520-0450(2003)042<0851:DCISHF>2.0.CO;2",
"Zhuang, Q. and B. Wu \
\"Estimating Evapotranspiration from an Improved Two-Source Energy \
Balance Model Using ASTER Satellite Imagery\" (2015) \
Water, 7(12), pp. 6673-6688. \
https://doi.org/10.3390/w7126653"]