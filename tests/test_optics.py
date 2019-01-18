from unittest import TestCase

from scimap import optics, exceptions


class ChemicalFormulaTestCase(TestCase):
    """Tests for parsing LiMn2O4 style formulas."""
    def test_bad_formula(self):
        with self.assertRaises(exceptions.ChemicalFormulaError):
            optics.parse_chemical_formula('hello')
    
    def test_simple_formula(self):
        result = optics.parse_chemical_formula('LiMn2O4')
        expected = [('Li', 1), ('Mn', 2), ('O', 4)]
        self.assertEqual(result, expected)
    
    def test_decimal_formula(self):
        formula = 'LiNi_{0.8}Co_{0.15}Al_{0.05}O2'
        result = optics.parse_chemical_formula(formula)
        expected = [('Li', 1), ('Ni', 0.8), ('Co', 0.15),
                    ('Al', 0.05), ('O', 2)]
        self.assertEqual(result, expected)


class MolarMassTestCase(TestCase):
    def test_single_element(self):
        self.assertEqual(optics.molar_mass('Li'), 6.941)
    
    def test_simple_formula(self):
        result = optics.molar_mass('LiMn2O4')
        self.assertEqual(result, 180.861)
    
    def test_decimal_formula(self):
        formula = 'LiNi_{0.8}Co_{0.15}Al_{0.05}O2'
        result = optics.molar_mass(formula)
        self.assertEqual(result, 96.0815)


class MassAttenuationCoefficientTestCase(TestCase):
    Cu_kalpha = 8047.8
    def test_simple_formula(self):
        mu = optics.mass_attenuation_coefficient('LiMn2O4', self.Cu_kalpha)
        self.assertAlmostEqual(mu, 167.7612, places=4)


class PhotoabsorptionCrossSectionTestCase(TestCase):
    Cu_kalpha = 8047.8
    def test_nickel(self):
        """Check the value reported at http://henke.lbl.gov/cgi-bin/pert_cgi.pl"""
        sigma = optics.photoabsorption_cross_section('Ni', self.Cu_kalpha)
        self.assertEqual(sigma, 46.69)


class TransmissionTest(TestCase):

    def test_nickel(self):
        """Test against real data, from
        http://henke.lbl.gov/optical_constants/filter2.html
        
        """
        real_transmission = 0.99171
        eV = 8047.8
        density = 8.902
        thickness_um = 0.2 # in microns
        thickness_cm = thickness_um * 1e-6 / 1e-2
        # Calculate the predicted transmission
        linear_coeff = density * optics.mass_attenuation_coefficient('Ni', eV)
        transmission = optics.transmission(distance=thickness_cm, linear_attenuation=linear_coeff)
        self.assertAlmostEqual(transmission, real_transmission, places=4)
