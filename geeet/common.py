"""Common utilities among modules and other functions"""

import sys, os, shutil

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