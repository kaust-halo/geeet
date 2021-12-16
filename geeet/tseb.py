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
                Leaf_width = 0.1, zU = None, zT = None
                ):
    """
    Priestley-Taylor Two Source Energy Balance Model
    Function to compute TSEB energy fluxes using a single observation
    of composite radiometric temperature using resistances in series (i.e. for dense vegetation). 

    Inputs:
    Either a single ee.Image (img) with the following inputs as bands (unless otherwise specified) 
    or inputs as numpy arrays (unless otherwise specified):

    - Radiometric temperature (Kelvin) e.g. Land surface temperature:   radiometric_temperature (band) or Tr (numpy array)
    - Normalized Difference Vegetation Index (NDVI): NDVI (band) or NDVI (numpy array)
    - Leaf Area Index (LAI; m2/m2): LAI (band) or LAI (numpy array)

    Scalar inputs:
    the following scalar inputs can be supplied either as a property in img or as parameters:
    - zU: Height of wind speed measurement (m) (or height of the "surface" level if data is from a climate model)
    - zT: Height of measurement of air temperature (m) (or height of the "surface" level if data is from a climate model)
    - leaf_width
    - CH: canopy height 
    - Leaf_width: 
    - AlphaPT: Priestly Taylor coefficient for canopy potential transpiration (defaults to 1.26)
    - F_g: fraction of vegetation that is green: default to 1.0

    """
    from geeet.vegetation import compute_lai, compute_ftheta, compute_Rns
    from geeet.solar import compute_solar_angles, compute_Rn, compute_g
    from geeet.meteo import compute_met_params, compute_roughness
    from geeet.resistances import RN95
    from geeet.MOST import Ustar as compute_ustar
    from geeet.MOST import MOL

    # Hard-coded scalar parameters
    max_iterations = 5 # for the main TSEB loop (LEs)
    # Note that only a few iterations is ok.. (empirically.. 5 is ok)
    # 5: it's ok.
    # 10: works. 
    # 100: computation is too complex.

    min_LAI = 1.0 # Threshold to use one-source (low LAI) or two-source energy balance (LAI > min_LAI)
    D_0_min = 0.004 # Minimum zero-plane displacement height (m).
    z_0M_min = 0.003  # Minimum value for the aerodynamic surface roughness length (m).

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
        from ee import Image
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

        # At this point, we have initial estimates of:
        # T (c, s) temperatures
        # LE (c, s) latent heat flux
        # H (c, s) sensible heat flux
        # resist (a,s,x) resistances
        # we now need to iteratively re-calculate all of these variables
        # but only for pixels where LEs remains negative. 

        # Currently we are using python to make the loop.
        # TODO: change this so that it is more GEE-friendly.
        # i.e., currently this is ok only with a few max_iterations (<10)
        # with larger numbers, it fails (computation too complex). 
        for iteration in range(max_iterations):
            # Restrict calculations to where LEs
            # is negative:
            pixelsToUpdate = LEs.lt(0) # mask indicating where we should continue processing in 
                                        # this iteration
            pixelsToRetain = LEs.gte(0) # mask indicating where we are NOT processing anymore. 

            # Split all variables that will be updated into
            # pixels to update (u) versus pixels to retain (no subscript).    
            LEsu = LEs.updateMask(pixelsToUpdate)
            LEcu = LEc.updateMask(pixelsToUpdate)
            LEu = LE.updateMask(pixelsToUpdate)
            Hcu = Hc.updateMask(pixelsToUpdate)
            Hsu = Hs.updateMask(pixelsToUpdate)
            Hu = H.updateMask(pixelsToUpdate)
            Tcu = Tc.updateMask(pixelsToUpdate)
            Tsu = Ts.updateMask(pixelsToUpdate)
            Tacu = Tac.updateMask(pixelsToUpdate)
            ustaru = ustar.updateMask(pixelsToUpdate)
            rau = ra.updateMask(pixelsToUpdate)
            rsu = rs.updateMask(pixelsToUpdate)
            rxu = rx.updateMask(pixelsToUpdate)

            LEs = LEs.updateMask(pixelsToRetain)
            LEc = LEc.updateMask(pixelsToRetain)
            LE = LE.updateMask(pixelsToRetain)
            Hc = Hc.updateMask(pixelsToRetain)
            Hs = Hs.updateMask(pixelsToRetain)
            H = H.updateMask(pixelsToRetain)
            Tc = Tc.updateMask(pixelsToRetain)
            Ts = Ts.updateMask(pixelsToRetain)
            Tac = Tac.updateMask(pixelsToRetain)
            ustar = ustar.updateMask(pixelsToRetain)
            ra = ra.updateMask(pixelsToRetain)
            rs = rs.updateMask(pixelsToRetain)
            rx = rx.updateMask(pixelsToRetain)

            #### Update stage:
            # In the following code, it is critical that 
            # any variable that is updated uses the "u" subscript
            # This includes further equations that
            # are used to update other variables!
             
            ### Update temperatures (Tc, Ts, Tac) using N95 equations (1, A.7-A.13)
            # Linear approximation of Tc (N95 A7):
            # Note that Hc = Rnc - LEc   where LEc = Rnc[alphaPT*Fg*taylor], so 
            # Hc = Rnc * (1 - alphaPT * Fg * taylor)
            Tclin_num1 = Ta.divide(rau)
            Tclin_num2 = Tr.divide(rsu.multiply(Image(1).subtract(f_theta)))
            Tclin_num3 = (Hcu.multiply(rxu).divide(rho.multiply(cp))).multiply((Image(1).divide(rau)).add(Image(1).divide(rsu)).add(Image(1).divide(rxu)))
            Tclin_denom = (rau.pow(-1)).add(rsu.pow(-1)).add(f_theta.divide(rsu.multiply(Image(1).subtract(f_theta))))
            Tclin = (Tclin_num1.add(Tclin_num2).add(Tclin_num3)).divide(Tclin_denom)

            # N95 equation A.12 (TD):
            Td1 = Tclin.multiply(Image(1).add(rsu.divide(rau)))
            Td2 = (Hcu.multiply(rxu).divide(rho.multiply(cp))).multiply(Image(1).add(rsu.divide(rxu)).add(rsu.divide(rau)))
            Td3 = Ta.multiply(rsu.divide(rau))
            Td = Td1.add(Td2).subtract(Td3)
            # N95 equation A.11 (deltaTc), i.e. correction to the linear approximation of Tc:
            dTc_num = (Tr.pow(4)).subtract(f_theta.multiply(Tclin.pow(4))).subtract((Image(1).subtract(f_theta)).multiply(Td.pow(4)))
            dTc_denom1 = (Td.pow(3)).multiply(rsu.divide(rau).add(1)).multiply(4).multiply(Image(1).subtract(f_theta))
            dTc_denom2 = f_theta.multiply(4).multiply(Tclin.pow(3))
            dTc = dTc_num.divide(dTc_denom1.add(dTc_denom2))
            # N95 equation A.13 (Tc = Tclin + delta Tc)
            Tcu = Tclin.add(dTc)

            # N95 equation 1, solving for (1-f(theta))*Ts^n  = Tr^n - f(theta)*Tc^n
            # the RHS of this equation is:
            TsRHS = Tr.pow(4).subtract(f_theta.multiply(Tcu.pow(4)))
            # Force TsRHS to be a positive value to avoid complex numbers:
            TsRHS = TsRHS.max(1e-6)
            # Estimate Ts (N95 equation 1)
            Tsu = (TsRHS.divide(Image(1).subtract(f_theta))).pow(0.25)
            Tsu = Tsu.max(1e-6)   # Force a minimum value for Ts 

            # Estimate Tac (N95 equation A.4):
            Tacu = ((Ta.divide(rau)).add(Tsu.divide(rsu)).add(Tcu.divide(rxu)))\
                    .divide((Image(1).divide(rau)).add(Image(1).divide(rsu)).add(Image(1).divide(rxu)))

            # Constraints on Tc and Ts based on DisALEXI
            Tsu = Tsu.where(f_theta.gte(0.9), Tr)
            Tsu = Tsu.where(f_theta.lte(0.1), Tr)
            Tcu = Tcu.where(f_theta.gte(0.9), Tr)
            Tcu = Tcu.where(f_theta.lte(0.1), Tr)

            ### Update sensible heat fluxes, now using the in-series network:
            Hcu = rho.multiply(cp).multiply(Tcu.subtract(Tacu)).divide(rxu)   # Tc: new; Tac: new; rx: old 
            Hsu = rho.multiply(cp).multiply(Tsu.subtract(Tacu)).divide(rsu)   # Tc: new; Tac: new; rs: old

            ### Update latent heat fluxes as a residual of the energy balance
            LEsu = Rns.subtract(G).subtract(Hsu) # Hs: new
            LEcu = Rnc.subtract(Hcu)             # Hc: new

            ### Update total fluxes
            LEu = LEcu.add(LEsu)                # LEc: new; LEs: new
            Hu = Hcu.add(Hsu)                   # Hc: new;  Hs: new
            
            ### Update M-O length (L) and friction velocity
            L = MOL(ustaru, Ta, rho, cp, Lambda, Hu, LEu)         #Ustar: old; H: new; LE: new 
            ustaru = compute_ustar(U, zU, L, rough_params = rough)  # L: new; 

            ### Update resistances (rau, rsu, rxu)
            # here we are calling RN95, but need to mask to ensure the resistances
            # are recalcuated only where needed. 
            resistU = RN95(U, CH, rough, LAI, Leaf_width, zU, zT, Ustar = ustaru, L=L)  # Ustar: new; L: new
            rau = resistU.select('Ra').updateMask(pixelsToUpdate)
            rsu = resistU.select('Rs').updateMask(pixelsToUpdate)
            rxu = resistU.select('Rx').updateMask(pixelsToUpdate)

            ### Update the PT constant (alpha):
            AlphaPT -= 0.01 
            AlphaPT = np.maximum(AlphaPT, 0)  # do not let it drop below 0

            # Finally, recompute LEc using the PT equation (N95 equation 12)
            # for any pixel that still has LEs<0:
            LEcu2 = taylor.multiply(Rnc).multiply(AlphaPT*F_g) 
            LEcu = LEcu.where(LEsu.lt(0), LEcu2)
            # and Hc = Rnc-LEc
            Hcu2 = Rnc.subtract(LEcu)
            Hcu = Hcu.where(LEsu.lt(0), Hcu2)

            ### Finally, update the actual variables by 
            # joining the "retained" and "updated" pixels:
            # note: unmask() removes the mask, and sets 0 everywhere. 
            # here LEs contains the retained values
            # LEsu contains the updated values. 
            # so adding them is ok (0+value | value+0)
            LEs = LEs.unmask().add(LEsu.unmask())
            LEc = LEc.unmask().add(LEcu.unmask())
            LE = LE.unmask().add(LEu.unmask())
            Hc = Hc.unmask().add(Hcu.unmask())
            Hs = Hs.unmask().add(Hsu.unmask())
            H = H.unmask().add(Hu.unmask())
            Tc = Tc.unmask().add(Tcu.unmask())
            Ts = Ts.unmask().add(Tsu.unmask())
            Tac = Tac.unmask().add(Tacu.unmask())
            ustar = ustar.unmask().add(ustaru.unmask())
            ra = ra.unmask().add(rau.unmask())
            rs = rs.unmask().add(rsu.unmask())
            rx = rx.unmask().add(rxu.unmask())
       
            # If no negative LEs remains, break iterations. 
            ######
        
        LE = LE.addBands(LEs).addBands(LEc).addBands(Ts).addBands(Tc).addBands(Hs)\
            .addBands(Hc).addBands(G).addBands(Rn).addBands(Rns).addBands(Rnc)\
                .addBands(ra).addBands(rs).addBands(rx)\
                    .rename(['LE', 'LEs','LEc', 'Ts', 'Tc','Hs','Hc','G','Rn','Rns','Rnc', 'Ra', 'Rs', 'Rx'])
        return img.addBands(LE)
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
        max_iterations=0
        for iteration in range(max_iterations):
            pixelsToUpdate = LEs<0 # in previous iteration, or from initialization
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

            if np.all(nLEs == False):
                break

        return LE, LEs, LEc, Hs, Hc, G, Rn, Rns, Rnc, Ts, Tc, ra, rs, rx