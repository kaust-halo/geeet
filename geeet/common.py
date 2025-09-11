"""Common utilities among modules and other functions"""

import sys, os, shutil
from typing import (Union, Optional)

def is_xr(obj):
    '''
    Function to check if an object is an instance of xarray.Dataset
    '''
    if 'xarray' in sys.modules:
        return isinstance(obj, sys.modules['xarray'].Dataset)
    else:
        return False


def is_img(obj):
    '''
    Function to check if an object is an instance of ee.Image
    '''
    if 'ee' in sys.modules:
        return isinstance(obj, sys.modules['ee'].Image)
    else:
        return False

def is_eenum(obj):
    '''
    Function to check if an object is an instance of ee.Number
    '''
    if 'ee' in sys.modules:
        return isinstance(obj, sys.modules['ee'].Number)
    else:
        return False


def update_package():
    """
    Updates the geeet package from the geeet GitHub repository main branch using pip.
    """
    try:
        if shutil.which("pip") is None:
            cmd = "pip3 install https://github.com/kaust-halo/geeet/archive/main.zip"
        else:
            cmd = "pip install https://github.com/kaust-halo/geeet/archive/main.zip"

        os.system(cmd)

    except Exception as e:
        raise Exception(e)


def find_spatial_dims(
    obj: object, relaxed: bool = False
    ) -> Optional[Tuple[str, str]]:
    """
    This function tries to replicate how odc.geo module finds spatial dimensions
    Find spatial dimensions of an xarray Dataset or DataArray.

    Checks for presence of dimensions named:
    ``y, x | latitude, longitude | lat, lon``

    If ``relaxed=True`` and none of the above dimension names are found,
    assume that last two dimensions are spatial dimensions.

    :returns: ``None`` if no dimensions with expected names are found
    :returns: ``('y', 'x') | ('latitude', 'longitude') | ('lat', 'lon')``
    """
    # dimensions with these names are considered spatial
    STANDARD_SPATIAL_DIMS = [
        ("y", "x"),
        ("yc", "xc"),
        ("latitude", "longitude"),
        ("lat", "lon"),
    ]
    
    def skip_dim(dim: str) -> bool:
        if dim in ("time", "band", "bands", "wavelength", "wavelengths"):
            return True

        # skip dimensions without coord of the same name
        if dim not in obj.coords:
            return True

        coord = obj.coords[dim]
        # Primary coordinate for spatial dimension must have floating point type
        if coord.dtype.kind != "f":
            return True

        return False

    if 'xarray' in sys.modules:
        if isinstance(obj, sys.modules['xarray'].Dataset) or isinstance(obj, sys.modules['xarray'].DataArray):
            _dims = [str(dim) for dim in obj.dims]
            dims = set(_dims)
            for guess in STANDARD_SPATIAL_DIMS:
                if dims.issuperset(guess):
                    return guess

            _dims = [dim for dim in _dims if not skip_dim(str(dim))]

            if relaxed and len(_dims) >= 2:
                return _dims[-2], _dims[-1]

            return None
    else:
        return None
