"""
Custom workflows 
"""

def masked_et(cfmaskable_bands,
               maskable_bands,
               positive_le = True,
               NDVI_BARE_GROUND = None,
               fvc = None
               ):
    """
    Custom ET estimation workflow

    1. Extrapolate LE (W/m²) -> ET (mm/day)
    2. Applies cloud-mask to cfmaskable_bands
    3. Adds the "feature_area" pixelArea band.
    4. Adds the "unobserved_area" pixelArea band.

    If positive_le is set to True:
        5. Adds a "positive_le_mask"
        6. Applies the "positive_le_mask" to maskable_bands
    
    If NDVI_BARE_GROUND is not None:
        7. Adds "vegetation_mask"
        8. Applies the "vegetation_mask" to maskable_bands
        9. Adds the "observed_vegetation_area" band.
        
    If fvc is not None:
        10. Adds the fractional vegetation cover (fvc) band. 
        11. Adds the "unobserved_vegetation_area" band. 
    """
    from . import landsat
    from . import pixel_area
    from . import masks

    cfmask = landsat.cfmask(cfmaskable_bands)
    lemask = masks.apply_static_mask("positive_le_mask", maskable_bands)
    vegmask = masks.apply_static_mask("vegetation_mask", maskable_bands)
    add_veg_mask = masks.Fndvi_mask(NDVI_BARE_GROUND)
    add_fvc = masks.Ffvc(fvc)

    workflow = [landsat.extrapolate_LE,
                cfmask,
                pixel_area.feature_area,
                pixel_area.unobserved_area
                ]
    
    if positive_le:
        workflow = workflow + [
            masks.positive_LE_mask, lemask
        ]

    if NDVI_BARE_GROUND:
        workflow = workflow + [
            add_veg_mask, vegmask,
            pixel_area.observed_veg_area
        ]    

    if fvc:
        workflow = workflow + [
            add_fvc, pixel_area.unobserved_veg_area, 
        ]

    return workflow