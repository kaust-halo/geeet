"""
Module for calculating heat transport resistances for energy balance models. 
"""
import numpy as np
from geeet.common import is_img

def RN95(U, CH, rough_params, LAI, leaf_width, zU, zT, L=None, Ustar = None, rough_bands = ['ZM','ZH','D0'] , band_names = ['Ra', 'Rs', 'Rx']):
    """
    Calculate the original TSEB resistances from Norman et al., 1995 (N95) 
#//description 
    Inputs:
        - U: wind speed in m/s, numpy array or an ee.Image
        - CH: canopy height in m, numpy array or ee.Image 
             (originally "hc" in N95, renamed CH here so that it is
             not confused with sensible heat flux)
        - rough_params: zm, zh, and d0, either as a list [] of numpy arrays,
                        or ee.Image (see geeet.meteo.compute_roughness) with three bands:
                        zm: roughness length for momentum transport (m)
                        zh: roughness length for heat transport     (m)
                        d0: zero-plane displacement height          (m)
        - LAI: leaf area index (m2/m2)  (originally "F" in N95). Numpy array or ee.Image
        Scalar inputs:
        - leaf_width: mean (or "effective") leaf width (m) 
                     (Goudriaan 1977; see equation B.3 in N95)
        - z_u: measurement height (m) for wind speed
        - z_t: measurement height (m) for temperature
        Optional:
        - L: Monin-Obukhov length (see geeet.MOST), defaults to a large value (neutral conditions)
        - Ustar: If Ustar is given, it is used in the calculation for RA. By default we use
                the explicit parameterization of N95 equation 6. 
        - rough_bands: (list) if rough_params is an ee.Image, rough_bands specifies the names
                       given to the roughness length for momentum, roughness length for
                       heat, and zero-plane displacement height, respectively. 
                       Defaults to the default values in geeet.resistances.compute_roughness,
                       i.e., 'ZM', 'ZH', and 'D0'
        - band_names: if calculating resistances as ee.Images and
                      band_names are given as a list of strings,
                      the bands are renamed before returning the output. 
                     
    Outputs (see diagrams below):
        - resistances: either a list of numpy arrays, or ee.Image with the following elements:
            - Ra: numpy array, or ee.Image with band name set to band_names[0]
            - Rs: numpy array, or ee.Image with band name set to band_names[1]
            - Rx: numpy array, or ee.Image with band name set to band_names[2]
   
                Dense canopy                        Sparse canopy
        In-series resistance network:        Parallel resistance network:
                              Tair                        Tair
                               |                            |          
                               /                     --------------
                               \ Ra                  |            |
                               /                     /            /
                       Rx      |                  Ra \            \ Ra
        Canopy   Tc---\/\/\/---| Tac        Canopy   |            |                
                               |                     Tc           |
                               /                                  /
                               \ Rs                               \ Rs
                               /                                  /
                               |                                  |
                               Ts                                 Ts
        ---------------------------------Soil-----------------------------
        Heat fluxes (not calculated here, just indicated for illustration purposes):
        Hc ~ (Tc-Tac)/Rx               |  (Tc-Tair)/Ra     (Ts-Tair)/(Ra+Rs)
        Hs ~ (Ts-Tac)/Rs               |            
        Hc+Hs = H ~ (Tac-Tair)/Ra      |     H ~ sum of above 2 terms
    (to transform ~ into =, scale by rho*Cp)

    Ra and Rs are both caculated equally for both networks.

    The parallel network allows a simpler computation and is ok for 
    sparse, clumped vegetation common to semi-arid regions*
    i.e. where canopy and soil interact less with each other

    Rx considers a "coupling" of the canopy/soil fluxes 

    *For irrigated agriculture with dense vegetation, e.g.
    center-pivots, the in-series network might be more appropriate. 

    Ra  is the aerodynamic resistance calculated from the diabatically corrected
        log temperature profile equations (Brutsaert, 1982)
        Ra = (0.16U)^-1  * [ln((zu-d)/zm) - psiM] * [ln((zt-d)/zm) - psiH]  (Equation 6)
            where zu, zt are measurement height of wind speed and temperature
            d~0.65*hc   (hc is canopy height)  displacement height
            zm~hc/8                            roughness length for momentum
    
    Rs  is the resistance to transport of heat between the soil surfce and a height
        representing the canopy.
        Rs = 1 / (a1 + b1 Us)   (Equation B1)
            where a1~0.004 ms-1; b1~0.012,
            Us is the wind speed at the soil surface (where the effect of the soil surface
            roughness is minimal) at about 0.05 m to 0.2 m
            and is related to the wind speed at the top of the canopy (Uc) (Equations B2 - B4)

    Rx  is unique to the in-series network (doesn't appear in the in-parallel)
        defined as the total boundary layer resistance of the complete canopy 
        of leaves
        Rx = (C1/LAI)(s/Udzm)^1/2  (Equation A.8)
        where C1 ~ 90 s-1/2 m-1
        s is the average leaf width
        Udzm is given by equation A.9 and is obtained after computing Uc (Equations B2-B4)
#//enddescription 
    """
    from geeet.MOST import PsiM as compute_psim
    from geeet.MOST import PsiH as compute_psih

    a1 = 0.004 # m s-1 ; originally a' in N95
    b1 = 0.012 # dimensionless? originally b' in N95
    C1 = 90  # s^-1/2 m-1 originally C' in N95

    karman = 0.4 # von Karman's constant

    if is_img(U):
        from ee import Image
        # Roughness parameters assumed to be bands
        # in an ee.Image (see geeet.meteo.rough_params):
        roughU = rough_params.select(rough_bands[0])
        roughT = rough_params.select(rough_bands[1])
        d0 = rough_params.select(rough_bands[2])

        # Monin-Obukhov length. 
        # If not provided, default to a large value (neutral conditions)
        # To avoid div/0, a minimum value is enforced.
        if L is None:
            L = Image(1e10)  

        if is_img(L):
            L = L.max(Image(1e-36))
        else:
            L = np.maximum(L, 1e-36) 
            L = Image(L)

        # Diabatic correction factors (PsiM and PsiH) for RA
        # (also called integration stability correction terms)
        # based on Brutsaert, 2005 
        # see geeet.MOS.psiM and geeet.MOS.psiH
        zU = Image(zU)
        zT = Image(zT)
        z_g = (zU.subtract(d0)).divide(L)
        PsiM = compute_psim(z_g)
        PsiMr = compute_psim(roughU.divide(L))
        PsiM = PsiM.subtract(PsiMr)   

        z_g_h = (zT.subtract(d0)).divide(L)
        PsiH = compute_psih(z_g_h)
        PsiHr = compute_psih(roughT.divide(L))
        PsiH = PsiH.subtract(PsiHr)

        # RA: Aerodynamic resistance (N95 equation 6)
        logM = ((zU.subtract(d0)).divide(roughU)).log()
        logT = ((zT.subtract(d0)).divide(roughT)).log() 
        #n.b. equation 6 in N95 contains an error
        # in the second log term: the numerator is 
        # indicated as "ZM" (here roughU) but 
        # it should be "ZH" (here roughT)
        if Ustar is None:
            # Explicitly compute N95 equation 6
            RA = (logM.subtract(PsiM)).multiply(logT.subtract(PsiH))
            RA = RA.divide(U.multiply(karman**2))                    
        else:
            # Compute N95 equation 6 with Ustar given as input:
            # (see geeet.MOST.Ustar)
            # Let Ustar = U*k / (ln((zU-d0)/roughU))-PsiM)
            #       RA = (ln((zU-d0)/roughU))-PsiM)*(ln((zT-d0)/roughT))-PsiH) / (U*k**2)
            #       RA = (ln((zU-d0)/roughU))-PsiM)*(1/(U*k))*(ln((zT-d0)/roughT))-PsiH) / k
            #       RA =                (1/Ustar)           *(ln((zT-d0)/roughT))-PsiH) / k  
            RA = (Image(1).divide(Ustar)).multiply(logT.subtract(PsiH)).divide(karman) 


        # Wind speed at top of canopy (Uc) (N95 Equation B.4)
        logC = ((CH.subtract(d0)).divide(roughU)).log()
        Uc = U.multiply(logC.divide(logM.subtract(PsiM))) 

        # "a" factor for Us by Goudriaan (1977) (N95 Equation B.3)
        a = (LAI.pow(2/3)).multiply(CH.pow(1/3)).multiply(0.28*leaf_width**(-1/3))

        # Wind speed just above soil surface (Us) (N95 Equation B.2)
        Us = ((((Image(0.2).divide(CH)).subtract(1)).multiply(a)).exp()).multiply(Uc)
        #               ^^
        # n.b. here we use 0.2 (N95 mentions 0.05m to 0.2m)
        # this should be the height above the soil surface
        # where the effect of the soil surface roughness
        # is minimal

        # RS: Soil resistance (N95 equation B.1)
        RS = Image(1).divide(Us.multiply(b1).add(a1))
        # ^^ End of computations for parallel network
        # Additional computations for in-series network (Rx):

        # Wind speed at height (d+zm) of momentum source sink (U_{d+zm}) (N95 Equation A.9)
        Udzm = ((((d0.add(roughU)).divide(CH)).subtract(1)).multiply(a)).exp().multiply(Uc)

        # RX: Resistance from canopy to canopy/soil space:
        RX = ((Image(leaf_width).divide(Udzm)).pow(1/2)).multiply(C1).divide(LAI)
        # n.b. equation A.8 in N95 contains an error:
        # the equation shows capital "S" (which would indicate
        # slope of the vapor pressure curve), but 
        # the text mentions small "s", which is leaf size
        # (here leaf_width) 

        resistances = (RA.addBands(RS).addBands(RX)).rename(band_names)
    else:
        # Roughness parameters should be provided as a list
        # in this order (see geeet.meteo.rough_params):
        roughU, roughT, d0 = rough_params

        # Monin-Obukhov length.
        # If not provided, default to a large value (neutral conditions)
        # To avoid div/0, a minimum value is enforced. 
        if L is None:
            L = 1e10 
        L = np.maximum(L, 1e-36) 

        # Diabatic correction factors (PsiM and PsiH) for RA
        # (also called integration stability correction terms)
        # based on Brutsaert, 2005 
        # see geeet.MOS.psiM and geeet.MOS.psiH
        z_g = (zU-d0)/L
        PsiM = compute_psim(z_g)
        PsiMr = compute_psim(roughU/L)
        PsiM = PsiM - PsiMr
        
        z_g_h = (zT-d0)/L
        PsiH = compute_psih(z_g_h)
        PsiHr = compute_psih(roughT/L)
        PsiH = PsiH - PsiHr

        # RA: Aerodynamic resistance (N95 equation 6)
        logM = np.log((zU - d0)/roughU)
        logT = np.log((zT - d0)/roughT)
        # n.b. the original equation in the paper most likely contains
        # an error: it shows ZM (here "roughU") in both log terms, but the second
        # should be ZH (here "roughT")        
        if Ustar is None:
            # Explicitly compute N95 equation 6
            RA = (logM-PsiM) * (logT-PsiH) / (U*(karman**2))
        else:
            # Compute N95 equation 6 with Ustar given as input:
            # (see geeet.MOST.Ustar)
            # Let Ustar = U*k / (logM-PsiM), where
            # logM and logT are defined as above
            #       RA = (logM-PsiM)*(logT-PsiH) / (U*k**2)
            #       RA = (logM-PsiM)*(1/(U*k))*(logH-PsiH) / k
            #       RA =      (1/Ustar)       *(logH-PsiH) / k  
            RA = (1/Ustar)*(logT-PsiH)/karman 

        # Wind speed at top of canopy (Uc) (N95 Equation B.4)
        Uc = U*(np.log((CH - d0)/roughU)/(logM-PsiM))  # B.4

        # "a" factor for Us by Goudriaan (1977) (N95 Equation B.3)
        a=0.28*LAI**(2/3.)*CH**(1/3.)*leaf_width**(-1/3.)

        # Wind speed just above the soil surface (N95 Equation B.2)
        Us = Uc*np.exp(-1*a*(1-(0.2/CH))) 
        #                       ^^
        # n.b. here we use 0.2 (N95 mentions 0.05m to 0.2m)
        # this should be the height above the soil surface
        # where the effect of the soil surface roughness
        # is minimal

        # RS: Soil resistance (N95 equation B.1)
        RS = 1/(a1+(b1*Us)) 

        # ^^ End of computations for parallel network
        # Additional computations for in-series network (Rx):

        # Wind speed at height (d+zm) of momentum source sink (U_{d+zm}) (N95 Equation A.9)
        Udzm=Uc*np.exp(a*(((d0+roughU)/CH)-1)) 
        # RX: Resistance from canopy to canopy/soil space (N95 equation A.8):
        RX = (C1/LAI)*((leaf_width/Udzm)**0.5)
        # n.b. equation A.8 in N95 contains an error:
        # the equation shows capital "S" (which would indicate
        # slope of the vapor pressure curve), but 
        # the text mentions small "s", which is leaf size
        # (here leaf_width) 
        resistances = [np.array(RA), np.array(RS), np.array(RX)] 

    return resistances
