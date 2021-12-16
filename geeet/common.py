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
    """Updates the geeet package from the geeet GitHub repository without the need to use pip or conda.
    In this way, I don't have to keep updating pypi and conda-forge with every minor update of the package.

    """
    try:
        download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        clone_repo(out_dir=download_dir)

        pkg_dir = os.path.join(download_dir, "geeet-main")
        work_dir = os.getcwd()
        os.chdir(pkg_dir)

        if shutil.which("pip") is None:
            cmd = "pip3 install ."
        else:
            cmd = "pip install ."

        os.system(cmd)
        os.chdir(work_dir)

        print(
            "\nPlease comment out 'geeet.update_package()' and restart the kernel to take effect:\nJupyter menu -> Kernel -> Restart & Clear Output"
        )

    except Exception as e:
        raise Exception(e)

def clone_repo(out_dir=".", unzip=True):
    """Clones the geeet GitHub repository.

    Args:
        out_dir (str, optional): Output folder for the repo. Defaults to '.'.
        unzip (bool, optional): Whether to unzip the repository. Defaults to True.
    """
    from geemap.common import download_from_url
    url = "https://github.com/kaust-halo/geeet/archive/main.zip"
    filename = "geeet-main.zip"
    download_from_url(url, out_file_name=filename, out_dir=out_dir, unzip=unzip)