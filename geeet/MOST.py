"""Functions related to Monin-Obukhov Similarity theory.
"""
from geeet.common import is_img
import numpy as np

karman = 0.4 # von Karman's constant
gravity = 9.8 # acceleration of gravity (m s-2)


def PsiM_stable(z_g):
    """ Calculate Psi function for momentum in MOS theory under stable conditions
    z_g = (z-d)/L 
    z_g stands for "lowercase Greek zeta" ( ζ ). 

    Inputs:
        - z_g (numpy array or ee.Image): dimensionless parameter z_g (i.e. ζ) 
        z_g should be >0 

    Outputs: 
        - psim (numpy array or ee.Image): value of the correction term Psim(ζ)  (dimensionless)
        Calculated using equation 2.59 from Brutsaert, 2005:
            Psim(z_g) = -a ln [z_g + (1+z_g^b)^(1/b)]
                    where a = 6.1 and b = 2.5 

    This function is intended to be used by the general PsiM function
    (which calls PsiM_stable and PsiM_unstable)
    and not directly by the user. 

    Brutsaert, W. (2005). Hydrology: An Introduction. Cambridge: Cambridge University Press. doi:10.1017/CBO9780511808470
    """
    a = 6.1  
    b = 2.5  
    if is_img(z_g):
        log_term = (((z_g.pow(b)).add(1)).pow(1/b)).add(z_g)
        psim = (log_term.log()).multiply(-a)
    else:
        if np.max(z_g)<0:
            print('This function only applies for stable conditions (z_g>=0)')
            return
        # ignore z<0 values:
        z_g = np.maximum(z_g,0)
        
        psim = -a*np.log(z_g + (1+z_g**b)**(1/b))
        
    return psim


def PsiH_stable(z_g):
    """ Calculate Psi function for heat in MOS theory under stable conditions
    z_g = (z-d)/L 
    z_g stands for "lowercase Greek zeta" ( ζ ). 

    Inputs:
        - z_g (numpy array or ee.Image): dimensionless parameter z_g (i.e. ζ) 
        z_g should be >0 

    Outputs: 
        - psih (numpy array or ee.Image): value of the correction term Psih(ζ)  (dimensionless)
        Calculated using equation 2.59 from Brutsaert, 2005:
            Psih(z_g) = -a ln [z_g + (1+z_g^b)^(1/b)]
                    where a = 6.1 and b = 2.5 
    This function is intended to be used by the general PsiH function
    (which calls PsiH_stable and PsiH_unstable)
    and not directly by the user. 

    Brutsaert, W. (2005). Hydrology: An Introduction. Cambridge: Cambridge University Press. doi:10.1017/CBO9780511808470
    """
    #N.b. this is identical to psim for stable conditions (see equation 2.58 and figure 2.11)
    return PsiM_stable(z_g)


def PsiM_unstable(z_g):
    """Calculate Psi function for momentum in MOS theory under unstable conditions (z_g<0)
    z_g stands for "lowercase Greek zeta" ( ζ ). 

    Inputs:
        - z_g (numpy array or ee.Image): dimensionless parameter z_g (i.e. ζ) 
        z_g should be <0 
    
    Outputs:
        - Psim (numpy array or ee.Image): value of the Psim(ζ) function (dimensionless)
        Calculated using equation 2.63 from Brutsaert, 2005:
            Psim(-y) =  ln(a+y) 
                         - 3b*(y^1/3)
                         + (b/2)*(a^1/3) ln (  (1+x)^2 / (1-x+x^2) )
                         + 3^(1/2)*b*(a^1/3) atan ((2x-1)/(3^1/2))
                         + Psi0                                         for y<=b^-3

            Psim(-y) = Psi_m (b^-3)                                    for y>b^-3
                         where x = (y/a)^(1/3) and y = -z_g = (-(z-d0)/L) 
                         and Psi0 = -ln a + 3^(1/2) b a^(1/3) pi / 6    
            
    This function is intended to be used by the general PsiM function
    (which calls PsiM_stable and PsiM_unstable)
    and not directly by the user. 

    Brutsaert, W. (2005). Hydrology: An Introduction. Cambridge: Cambridge University Press. doi:10.1017/CBO9780511808470
    """
    a = 0.33
    b = 0.41

    Psi_0 = -np.log(a) + 3**(1/2)*b*a**(1/3)*np.pi/6  # ~1.36

    if is_img(z_g):
        from ee import Image

        # Ignore z>0 values (however note that PsiM function will filter these out.):
        z_g = z_g.min(Image(0))
        y = (z_g.multiply(-1)).min(Image(b**-3)) # Max value for y
        x = (y.divide(a)).pow(1/3)
        psim1 = (y.add(a)).log()
        psim2 = (y.pow(1/3)).multiply(-3*b)
        psim3_lnTop = ((x.add(1)).pow(2))
        psim3_lnBottom = (x.pow(2)).add(1).subtract(x)
        psim3 = (psim3_lnTop.divide(psim3_lnBottom)).log().multiply((b/2)*(a**(1/3)))
        psim4_tanArg = ((x.multiply(2)).subtract(1)).divide(3**(1/2))
        psim4 = (psim4_tanArg.atan()).multiply((3**(1/2))*b*(a**(1/3)))

        psim = psim1.add(psim2).add(psim3).add(psim4).add(Psi_0)
       
    else:
        # Check z (only applicable for z<0)
        if np.min(z_g)>=0:
            print('This function only applies for unstable conditions (z_g <0)')
            return
        # Ignore z>0 values:
        z_g = np.minimum(z_g,0)
        y = -z_g
        y = np.minimum(y, b**-3)  # max y is ~14.56 (i.e. z lowest value is ~-14.56 --> PsiM max is ~1.8)
        x = (y/a)**(1/3)
        psim = np.log(a+y)\
               - 3*b*(y**(1/3))\
               + (b/2)*(a**(1/3))*np.log((1+x)**2 / (1-x+x**2))\
               + (3**(1/2))*b*(a**(1/3))*np.arctan((2*x-1)/(3**(1/2)))\
               + Psi_0
    return psim


def PsiH_unstable(z_g):
    """Calculate Psi function for heat in MOS theory under unstable conditions (z_g<0)
    z_g stands for "lowercase Greek zeta" ( ζ ). 

    Inputs:
        - z_g (numpy array or ee.Image): dimensionless parameter z_g (i.e. ζ) 
        z_g should be <0 
    
    Outputs:
        - Psih (numpy array or ee.Image): value of the Psih(ζ) function (dimensionless)
        Calculated using equation 2.64 from Brutsaert, 2005:
            Psih(-y) =  [(1-d)/n]*ln[(c+y^n)/c]

    This function is intended to be used by the general PsiH function
    (which calls PsiH_stable and PsiH_unstable)
    and not directly by the user. 

    Brutsaert, W. (2005). Hydrology: An Introduction. Cambridge: Cambridge University Press. doi:10.1017/CBO9780511808470
    """
    c = 0.33
    d = 0.057
    n = 0.78
    dn = (1-d)/n
    if is_img(z_g):
        from ee import Image

        # Ignore z>0 values (however note that PsiH function will filter these out.):
        z_g = z_g.min(Image(0))
        y = (z_g.multiply(-1))

        psih = (y.pow(n).add(c)).divide(c).log().multiply(dn)
    else:
        # Check z (only applicable for z<0)
        if np.min(z_g)>=0:
            print('This function only applies for unstable conditions (z_g <0)')
            return
        # Ignore z>0 values:
        z_g = np.minimum(z_g,0)
        y = -z_g
        psih = dn*np.log((c+y**n)/c)

    return psih


def PsiM(z_g):
    """
    Calculate the diabatic correction factor (also called integrated stability correction term)
    for momentum (Psim) in MOS theory
    z_g stands for "lowercase Greek zeta" ( ζ ). 

    Inputs:
        - z_g (numpy array or ee.Image): dimensionless parameter z_g (i.e. ζ) 
    
    This function uses PsiM_stable and PsiM_unstable (defined above).

    Outputs: 
        - Psim (numpy array or ee.Image): value of the Psim(ζ) function (dimensionless)
    """
    if is_img(z_g):
        ZStable = z_g.where(z_g.lt(0), -1)  # Negative pixels -> -1
        ZStable = ZStable.updateMask(ZStable.gte(0)) # Keep only positive pixels

        ZUnstable = z_g.where(z_g.gt(0), 1) # Positive pixels -> 1
        ZUnstable = ZUnstable.updateMask(ZUnstable.lt(0)) #Keep only negative pixels

        PsimStable = PsiM_stable(ZStable)
        PsimUnstable = PsiM_unstable(ZUnstable)

        Psim= PsimStable.unmask().add(PsimUnstable.unmask()) 
        Psim = Psim.updateMask(z_g.mask()) # finally, apply the original mask. 
    else:
        if np.max(z_g)<0:
            Psim = PsiM_unstable(z_g)
        elif np.min(z_g)>=0:
            Psim = PsiM_stable(z_g)
        else:
            stable = z_g>=0
            unstable = z_g<0
            Psim = np.zeros_like(z_g)
            Psim[stable] = PsiM_stable(z_g[stable])
            Psim[unstable] = PsiM_unstable(z_g[unstable])

    return Psim


def PsiH(z_g):
    """
    Calculate the diabatic correction factor (also called integrated stability correction term)
    for heat (Psih) in MOS theory
    z_g stands for "lowercase Greek zeta" ( ζ ). 

    Inputs:
        - z_g (numpy array or ee.Image): dimensionless parameter z_g (i.e. ζ) 
    
    This function uses PsiH_stable and PsiH_unstable (defined above).
    n.b. the stable function is identical to the momentum one. 
    i.e., PsiH_stable = PsiM_stable

    Outputs: 
        - Psih (numpy array or ee.Image): value of the Psih(ζ) function (dimensionless)
    """
    if is_img(z_g):
        ZStable = z_g.where(z_g.lt(0), -1)  # Negative pixels -> -1
        ZStable = ZStable.updateMask(ZStable.gte(0)) # Keep only positive pixels

        ZUnstable = z_g.where(z_g.gt(0), 1) # Positive pixels -> 1
        ZUnstable = ZUnstable.updateMask(ZUnstable.lt(0)) #Keep only negative pixels

        PsihStable = PsiH_stable(ZStable)
        PsihUnstable = PsiH_unstable(ZUnstable)

        Psih= PsihStable.unmask().add(PsihUnstable.unmask()) 
        Psih = Psih.updateMask(z_g.mask()) # finally, apply the original mask. 
    else:
        if np.max(z_g)<0:
            Psih = PsiH_unstable(z_g)
        elif np.min(z_g)>=0:
            Psih = PsiH_stable(z_g)
        else:
            stable = z_g>=0
            unstable = z_g<0
            Psih = np.zeros_like(z_g)
            Psih[stable] = PsiH_stable(z_g[stable])
            Psih[unstable] = PsiH_unstable(z_g[unstable])

    return Psih

def Ustar(U, zU, L=None, rough_params=None, rough_bands = ['ZM','ZH','D0'], band_name='Ustar', minValue = 0.35):
    """
    Calculate friction velocity (U*)
    u_star = k*U/  (   ln(ζ) - Psim(ζ) - Psim(Z0m/L)  )
    where ζ (or z_g "z greek" ) = (z-d0)/L (dimensionless) is the MO stability parameter, 
    Z0m is the aerodynamic surface surface roughness length (m), 
    d0 is the zero-plane displacement height (m), and
    L is the Monin-Obukhov length
    
    Eq. 2.54 in Brutsaert, 2005. 

    Inputs:
        - U (numpy array or ee.Image): wind speed (m/s)
        - zU (scalar, float): height of wind speed measurement in meters (e.g. 2m for some numerical weather prediction models)
        - L (scalar, numpy array, or ee.Image): Monin-Obukhov length (m). Defaults to a high number for stable conditions. 
        - rough_params (list or ee.Image):
            Z_0M: aerodynamic surface roughness length for momentum transfer (m)
            Z_0H: aerodynamic surface roughness length for heat transfer (m) (not used)
            D_0: zero-plane displacement height (m)
            if rough_params is a list, they are retrieved as:
                Z_0M, _, D_0 = rough_params  
            if rough_params is an ee.Image, they are retrieved using ee.Image.select('band_name') where band_name is
                the first (for Z0) and second (for D_0) elements of the string list in rough_bands. 
                See: geeet.resistances.compute_roughness
        - rough_bands (optional, list): The first element is used as the band name to select the Z_0M band in rough_params 
                                       The second element is used as the band name to select the D_0 band in rough_params
                                       Defaults to ['Z0', 'D0'] (see: geeet.resistances.compute_roughness)
        - minValue (optional, scalar): Enforce a minimum value for U*, defaults to 0.35
                     RA is inversely proportional to U* = U*k/ (ln ((zu-d)/ZM) - PsiM)
                     Setting a minimum value for Ustar is equivalent to constraining 
                     RA to a maximum value:
                     Let lnM = ln((zu-d)/ZM)) - PsiM; lnH = ln((zt-d)/ZH) - PsiH. Then:
                         RA = lnM lnH / U k^2 
                         RA = lnM / (Uk)  lnH / k
                         RA = (1/U*) lnH/k
                         Constraining U* to a minimum value will constrain
                         RA to a maximum value of RA = (1/min_Ustar)*lnH/k 

    Outputs: 
        - u_star (numpy array or ee.Image): the friction velocity 
    """
    from geeet.MOST import PsiM as compute_psim

    karman = 0.4 # von Karman's constant

    if is_img(U):
        from ee import Image
        # Roughness parameters assumed to be bands
        # in an ee.Image (see geeet.meteo.rough_params):
        roughU = rough_params.select(rough_bands[0])
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
        z_g = (zU.subtract(d0)).divide(L)
        PsiM = compute_psim(z_g)
        PsiMr = compute_psim(roughU.divide(L))
        PsiM = PsiM.subtract(PsiMr)   

        logM = ((zU.subtract(d0)).divide(roughU)).log()
        u_star = (U.multiply(karman)).divide(logM.subtract(PsiM))
        u_star = u_star.rename(band_name)
        u_star = u_star.max(minValue)

    else:
        # Roughness parameters should be provided as a list
        # in this order (see geeet.meteo.rough_params):
        roughU, _, d0 = rough_params

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
        
        u_star = U*karman/((np.log((zU - d0)/roughU) - PsiM ))
        u_star = np.maximum(u_star, minValue)
        u_star = np.array(u_star)
    return u_star


def MOL(u_star, Ta_K, rho, cp, Lambda, H, LE):
    """
    Calculate the Monin-Obukhov stability length (L)
    L = -u*^3 /  ( k*(g/Ta)*Hv/(rho*cp))
    where u* is the friction velocity
    and Hv= H + (0.61*Ta*cp*E) 
    Eq. 2.46 in Brutsaert, 2005. 
    Inputs:  either all inputs should be numpy arrays, or all should be ee.Images:
        - u_star (numpy array or ee.Image): friction velocity, in m/s
        - Ta_K (numpy array or ee.Image): air tempearture, in K 
        - rho (numpy array or ee.Image): air density, in kg m-3
        - cp (numpy array or ee.Image): air heat capacity, in J kg-1 K-1
        - Lambda (numpy array or ee.Image): latent heat of vaporization, in MJ kg-1
            (n.b. to compute rho, cp, Lambda, see geeet.meteo.compute_met_params)
        - H (numpy array or ee.Image): sensible heat flux, in W m-2
        - LE (numpy array or ee.Image): latent heat flux, in W m-2
    Outputs: 
        - L (numpy array or ee.Image): Monin-Obukhov length (m)

    """
    if is_img(u_star):  # assuming all inputs are ee.Image
        from ee import Image
        E = LE.divide(Lambda.multiply(1e6)) # kg m-2 s-1
        Hv = H.add(Ta_K.multiply(cp).multiply(E).multiply(0.61))
        Hv = Hv.where(Hv.lte(0),Image(1e-10)) # avoiding div/0
        L = (u_star.pow(3)).multiply(-1.0).multiply(cp).multiply(rho).multiply(Ta_K).divide(karman*gravity)
        L = L.divide(Hv)
    else:
        E = LE/(Lambda*1e6)
        Hv = H + (0.61*Ta_K*cp*E)
        Hv = np.array(Hv)
        Hv[Hv<=0] = 1e-10 # avoiding div/0
        L = -u_star**3 / (Hv*karman*gravity/Ta_K*rho*cp)
        L = np.array(L)
        L[np.isnan(L)]=1e10 # replace nans by a large value. 
    return L