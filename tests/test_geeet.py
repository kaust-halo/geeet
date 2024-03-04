#!/usr/bin/env python

"""Tests for `geeet` package."""


import unittest
import xarray as xr

from geeet import geeet


class TestGeeet(unittest.TestCase):
    """Tests for `geeet` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        import numpy as np
        self.tseb_series_inputs = dict(
            Tr = [295, 295],
            Alb = [0.2, 0.2],
            NDVI = [0.8, 0.8],
            P = [95500, 95500],
            Ta = [293, 293], 
            U = [5,5], 
            Sdn = [800, 400], 
            Ldn = [300, 200]
        )
        self.tseb_series_inputs_scalar = dict(
            doy = 1,
            time = 11,
            Vza = 0,
            longitude = 38.25,
            latitude = 30,
            zU = 10,
            zT = 2
        )
        self.tseb_series_list = geeet.tseb.tseb_series(
            **self.tseb_series_inputs,
            **self.tseb_series_inputs_scalar
        )
        self.tseb_series_ndarray = geeet.tseb.tseb_series(
            **{key:np.array(value) for key,value in self.tseb_series_inputs.items()},
            **self.tseb_series_inputs_scalar
        )
        #self.tseb_series_inputs_xr=xr.merge([
        #        xr.DataArray(self.tseb_series_inputs["Alb"]).rename("albedo"),
        #        xr.DataArray(self.tseb_series_inputs["NDVI"]).rename("NDVI"),
        #        xr.DataArray(self.tseb_series_inputs["Tr"]).rename("radiometric_temperature"),
        #        xr.DataArray(self.tseb_series_inputs["Ta"]).rename("air_temperature"),
        #        xr.DataArray(self.tseb_series_inputs["P"]).rename("surface_pressure"),
        #        xr.DataArray(self.tseb_series_inputs["U"]).rename("wind_speed"),
        #        xr.DataArray(self.tseb_series_inputs["Sdn"]).rename("solar_radiation"),
        #        xr.DataArray(self.tseb_series_inputs["Ldn"]).rename("thermal_radiation"),
        #    ])            
        #self.tseb_series_xr = geeet.tseb.tseb_series(
        #    self.tseb_series_inputs_xr, **self.tseb_series_inputs_scalar
        #)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_tseb_series_list_consistency(self):
        """Test tseb_series outputs consistency (list)."""
        self.assertCountEqual(self.tseb_series_list["LE"] -
            self.tseb_series_list["LEs"] - self.tseb_series_list["LEc"],
            [0,0]
        )
        self.assertCountEqual(self.tseb_series_list["Rn"]-
            self.tseb_series_list["Rns"] - self.tseb_series_list["Rnc"],
            [0,0]
        )

    def assertListAlmostEqual(self, list1, list2, tol):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
             self.assertAlmostEqual(a, b, tol)

    def test_tseb_series_list_energy_balance(self):
        """Test tseb_series energy balance (list)."""
        self.assertListAlmostEqual(self.tseb_series_list["Rn"]-
            self.tseb_series_list["LEc"] -
            self.tseb_series_list["LEs"] -
            self.tseb_series_list["G"] - 
            self.tseb_series_list["Hc"] - 
            self.tseb_series_list["Hs"],
            [0,0],
            10
        )

    def test_tseb_series_np_consistency(self):
        """Test tseb_series outputs consistency (np.array)."""
        self.assertCountEqual(self.tseb_series_ndarray["LE"] -
            self.tseb_series_ndarray["LEs"] - self.tseb_series_ndarray["LEc"],
            [0,0]
        )
        self.assertCountEqual(self.tseb_series_ndarray["Rn"] -
            self.tseb_series_ndarray["Rns"] - self.tseb_series_ndarray["Rnc"],
            [0,0]
        )

    def test_tseb_series_np_energy_balance(self):
        """Test tseb_series energy balance (ndarray)."""
        self.assertListAlmostEqual(self.tseb_series_ndarray["Rn"]-
            self.tseb_series_ndarray["LEc"] -
            self.tseb_series_ndarray["LEs"] -
            self.tseb_series_ndarray["G"] - 
            self.tseb_series_ndarray["Hc"] - 
            self.tseb_series_ndarray["Hs"],
            [0,0],
            10
        )

    #def test_tseb_series_xr_consistency(self):
    #    """Test tseb_series outputs consistency (xr.DataArray)."""
    #    residuals = self.tseb_series_xr.assign(
    #            LEr = lambda x: x.LE-x.LEc-x.LEs,
    #            Rnr = lambda x: x.Rn-x.Rnc-x.Rns
    #    )
    #    self.assertListAlmostEqual(
    #        residuals.LEr.values,
    #        [0,0], 10
    #    )
    #    self.assertListAlmostEqual(
    #        residuals.Rnr.values,
    #        [0,0], 10
    #    )

    #def test_tseb_series_xr_energy_balance(self):
    #    """Test tseb_series energy balance (xr.DataArray)."""
    #    self.assertListAlmostEqual(
    #        self.tseb_series_xr.assign(
    #            EB = lambda x: x.Rn-x.LEc-x.LEs-x.G-x.Hc-x.Hs,
    #    ).EB.values,
    #        [0,0], 10
    #    )

if __name__ == "__main__":
    unittest.main()