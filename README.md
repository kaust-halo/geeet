# geeet

[![image](https://img.shields.io/pypi/v/geeet.svg)](https://pypi.python.org/pypi/geeet)
[![image](https://static.pepy.tech/badge/geeet)](https://pepy.tech/project/geeet)
[![image](https://img.shields.io/conda/vn/conda-forge/geeet.svg)](https://anaconda.org/conda-forge/geeet)
![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/geeet.svg)
[![image](https://github.com/kaust-halo/geeet/workflows/Linux%20build/badge.svg)](https://github.com/kaust-halo/geeet/actions)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Evapotranspiration (ET) models for use in python and with integration into Google Earth Engine.**

*geeet* provides hybrid evapotranspiration (ET) models that work with numerical values and with Google Earth Engine images.

- GitHub repo: https://github.com/kaust-halo/geeet
- PyPI: https://pypi.org/project/geeet/
- Conda-forge: https://anaconda.org/conda-forge/geeet
- Free software: MIT license

Inputs to geeet models can be given as [numpy arrays](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html), an [xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html), or a [ee.Image](https://developers.google.com/earth-engine/apidocs/ee-image).

## Introduction

*geeet* is a Python package providing a common set of building blocks for estimating evapotranspiration (ET) from remote sensing observations. It also features complete ET models such as [PT-JPL](https://doi.org/10.1016/j.rse.2007.06.025) and [TSEB](https://doi.org/10.1016/0168-1923(95)02265-Y). All modules in *geeet* are designed to work with input data provided in two formats: (1) as [numpy ndarrays](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) and (2) as [Google Earth Engine](https://earthengine.google.com/) (GEE) [images](https://developers.google.com/earth-engine/apidocs/ee-image). GEE is a cloud-based platform dedicated to Earth observation research that offers a [multi-petabyte catalogue](https://developers.google.com/earth-engine/datasets/) of geospatial data. Importantly, GEE offers cloud computing capabilities, which means that a user can interact with this geospatial data directly without having to download or process any data on premises. Access to these cloud-based services requires [signing up for a GEE account](https://earthengine.google.com/signup/). The *geeet* Python package was created to offer ET modeling tools that work for any user, whether they have a GEE account or not. For this reason, numpy is the only requirement for *geeet*.

> Integration of geeet with xarray and especially lazy evaluation using dask is currently under development, with the goal of estimating ET using cloud-based assets (e.g., Cloud Optimized Geotiffs through a Spatio Temporal Asset Catalog (STAC)).

## Installation

*geeet* is available on the [Python Package Index (pip)](https://pypi.org/project/geeet/). To install *geeet* using pip, run this command:

```bash
pip install geeet
```

If you want to install the latest development version hosted on github, run this command:

```bash
pip install git+https://github.com/kaust-halo/geeet
```

*geeet* is also available on [conda-forge](https://anaconda.org/conda-forge/geeet). To install *geeet* using conda, run this command:

```bash
conda install -c conda-forge geeet
```

The only requirement is a modern python installation (e.g. >3.7) and [numpy](https://numpy.org/). However, to test any of the *GEE* capabilities of *geeet* you will need to install the Python earthengine API (available through [pip](https://pypi.org/project/earthengine-api/) and [conda](https://anaconda.org/conda-forge/earthengine-api)), and have a [GEE account](https://earthengine.google.com/signup/).

## Quick start

If you have a GEE account and the earthengine API installed, we recommend first taking a look at [this notebook](https://github.com/kaust-halo/geeet/blob/main/examples/notebooks/01_geeet.ipynb) demonstrating the basic use of the hybrid ET models with a simple toy example. In a nutshell, running one of the pre-built models can be done in two lines of code, e.g.:

```python
from geeet.tseb import import tseb_series
et_tseb = tseb_series(img = sample_tseb_inputs) 
```

where `sample_tseb_inputs` is either a `ee.Image` or `xarray.Dataset` containing all the necessary inputs for the TSEB model.

*geeet* models can also be mapped to an [`ee.ImageCollection`](https://developers.google.com/earth-engine/guides/ic_creating), e.g.:

```python
from geeet.ptjpl import ptjpl_arid
et_outputs = et_inputs.map(ptjpl_arid)
```

where `et_inputs` is an `ee.ImageCollection` with the required inputs.

## PT-JPL model for arid environments (as described in [Aragon et al., 2018](http://dx.doi.org/10.3390/rs10121867))

This [notebook](./examples/notebooks/02_demo_using_GEE_data.ipynb) is a self-contained example that demonstrates the use of real GEE datasets with this PT-JPL model.

You can preview a pre-processed output of this example [here](https://code.earthengine.google.com/?scriptPath=users%2Flopezvoliver%2Fgeeet%3Aptjpl_sample_outputs_coarse) (requires a GEE account).  

## Two-source Energy Balance model ([TSEB](https://doi.org/10.1016/0168-1923(95)02265-Y))

*geeet* includes a two-source energy balance model mostly based on the original parameterizations of [Norman et al., 1995](https://doi.org/10.1016/0168-1923(95)02265-Y). Specifically, it initializes the estimates of the temperatures of the soil and the canopy layers using a Priestley-Taylor equation. It then iteratively updates the temperatures, energy fluxes, and resistance values using the in-series resistance network parameterization.

A pre-defined TSEB model with [Landsat](https://github.com/kaust-halo/geeet/blob/main/examples/notebooks/03_eepredefined_landsat_era5.ipynb) images and [ERA5 climate reanalysis data](https://github.com/kaust-halo/geeet/blob/main/examples/notebooks/03_eepredefined_landsat_era5.ipynb) is also available. Learn how to use this model [here](./examples/notebooks/03_eepredefined_landsat_era5.ipynb), specifically:

1. Prepare a merged Landsat collection (Landsat 7, 8, and 9)
2. Prepare a joint Landsat + ERA5 image collection
3. Map TSEB onto the Landsat+ERA5 collection (see also [this example](./examples/notebooks/04_eepredefined_landsat_mapped_collection.ipynb))

### Xarray and COG support

This [example](./examples/notebooks/05_xarray_landsat_era5.ipynb) demonstrates reading a Cloud-Optimized Geotiff (COG) using [rioxarray](https://corteva.github.io/rioxarray/stable/) and running the same TSEB model with this image.

## References

References for each model are found in [REFERENCES.txt](REFERENCES.txt). The source code for each module contains references for each function as well. Finally, each model contains two functions to display the references: `cite()` shows the main citation for the model, while `cite_all()` shows all the references for that model.

If you use this package for research, please cite the relevant model references. 

## Contributions

Contributions are welcome. Please [open an issue](https://github.com/kaust-halo/geeet/issues) to:

- suggest a relevant ET model
- report a bug
- report an issue

Feel free to submit a [pull request](https://github.com/kaust-halo/geeet/pulls) for suggesting code improvements.

## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
