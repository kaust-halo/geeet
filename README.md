# geeet


[![image](https://img.shields.io/pypi/v/geeet.svg)](https://pypi.python.org/pypi/geeet)


**Evapotranspiration (ET) models for use in python and with integration into Google Earth Engine.**

geeet aims to provide users with ready-to-use evapotranspiration (ET) models, both for numerical values and for use with Google Earth Engine. 

This initial release features a PT-JPL model adapted for arid environments (as described in Aragon et al., 2018). A notebook example is included in `./examples/notebooks/01_PTJPL.ipynb`. 

-   Free software: MIT license
-   Documentation: https://kaust-halo.github.io/geeet


## Features

- PT-JPL model for arid environments (as described in Aragon et al., 2018)    

## Installation

`pip install geeet`

A release will soon be uploaded to conda-forge. 

## References

References for each model are found in [REFERENCES.txt](REFERENCES.txt). The source code for each module contains references for each function as well. Finally, each model contains two functions to display the references: `cite()` shows the main citation for the model, while `cite_all()` shows all the references for that model.

If you use this package for research, please cite the relevant model. 

## Contributions

Contributions are welcome. We aim to include as many ET models to allow researchers to intercompare the different models. 

## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
