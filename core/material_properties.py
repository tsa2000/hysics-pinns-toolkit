#!/usr/bin/env python3
"""
Material Properties Calculator

Calculate thermodynamic and transport properties using CoolProp and Cantera.
Provides material data for PINNs boundary conditions and source terms.

Dependencies: CoolProp, cantera, numpy
Usage: python material_properties.py --fluid air --temp 300 --pressure 101325
"""

import numpy as np
from typing import Dict, List, Optional
import warnings
import argparse

# Try importing CoolProp and Cantera
try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    print("Warning: CoolProp not available. Using mock data.")
    COOLPROP_AVAILABLE = False

try:
    import cantera as ct
    CANTERA_AVAILABLE = True
except ImportError:
    print("Warning: Cantera not available. Using mock data.")
    CANTERA_AVAILABLE = False


class MaterialProperties:
    """Calculate material properties for PINNs applications"""

    def __init__(self):
        self.coolprop_available = COOLPROP_AVAILABLE
        self.cantera_available = CANTERA_AVAILABLE

        # Common fluid names mapping
        self.fluid_map = {
            'air': 'Air.mix',
            'water': 'Water',
            'nitrogen': 'Nitrogen',
            'oxygen': 'Oxygen',
            'hydrogen': 'Hydrogen',
            'methane': 'Methane',
            'propane': 'Propane',
            'co2': 'CarbonDioxide',
            'co': 'CarbonMonoxide'
        }

    def get_fluid_properties(self, fluid: str, temperature: float,
                             pressure: float, properties: Optional[List[str]] = None) -> Dict:
        """
        Get fluid properties using CoolProp or mock data.

        Args:
            fluid: Fluid name
            temperature: Temperature in K
            pressure: Pressure in Pa
            properties: List of properties to calculate

        Returns:
            Dictionary of properties
        """
        if properties is None:
            properties = ['density', 'viscosity', 'thermal_conductivity',
                          'specific_heat', 'enthalpy', 'entropy']

        if not self.coolprop_available:
            return self._get_mock_fluid_properties(fluid, temperature, pressure, properties)

        try:
            fluid_name = self.fluid_map.get(fluid.lower(), fluid)
            results = {}

            # Map property names to CoolProp names
            cp_prop_map = {
                'density': 'D',
                'viscosity': 'V',
                'thermal_conductivity': 'L',
                'specific_heat': 'C',
                'enthalpy': 'H',
                'entropy': 'S',
                'speed_of_sound': 'A',
                'compressibility': 'Z'
            }

            for prop in properties:
                if prop in cp_prop_map:
                    try:
                        value = CP.PropsSI(cp_prop_map[prop], 'T', temperature,
                                          'P', pressure, fluid_name)
                        results[prop] = value
                    except Exception as e:
                        print(f"Warning: Could not calculate {prop} for {fluid}: {e}")
                        results[prop] = self._get_default_property(prop, fluid)
                else:
                    print(f"Warning: Unknown property {prop}")

            # Add derived properties
            if 'density' in results and 'viscosity' in results:
                results['kinematic_viscosity'] = results['viscosity'] / results['density']

            if ('thermal_conductivity' in results and 'specific_heat' in results
                    and 'density' in results):
                results['thermal_diffusivity'] = (results['thermal_conductivity'] /
                                                 (results['density'] * results['specific_heat']))

            return results

        except Exception as e:
            print(f"Error calculating properties for {fluid}: {e}")
            return self._get_mock_fluid_properties(fluid, temperature, pressure, properties)

    def _get_mock_fluid_properties(self, fluid: str, temperature: float,
                                  pressure: float, properties: List[str]) -> Dict:
        """Generate mock fluid properties for testing"""
        # Typical values for air at standard conditions
        mock_values = {
            'density': 1.225 * (pressure / 101325) * (273.15 / temperature),
            'viscosity': 1.81e-5 * (temperature / 273.15)**0.7,
            'thermal_conductivity': 0.024 * (temperature / 273.15)**0.8,
            'specific_heat': 1005.0,
            'enthalpy': 1005.0 * temperature,
            'entropy': 1005.0 * np.log(temperature / 273.15),
            'speed_of_sound': 343.0 * np.sqrt(temperature / 273.15),
            'compressibility': pressure / (287.0 * temperature * 1.225)
        }

        # Adjust for different fluids
        fluid_factors = {
            'water': {'density': 1000, 'viscosity': 1e-3, 'thermal_conductivity': 0.6},
            'hydrogen': {'density': 0.08, 'viscosity': 8.8e-6, 'thermal_conductivity': 0.18},
            'methane': {'density': 0.7, 'viscosity': 1.1e-5, 'thermal_conductivity': 0.033}
        }

        if fluid.lower() in fluid_factors:
            for prop, factor in fluid_factors[fluid.lower()].items():
                if prop in mock_values:
                    mock_values[prop] = factor

        # Calculate derived properties
        mock_values['kinematic_viscosity'] = mock_values['viscosity'] / mock_values['density']
        mock_values['thermal_diffusivity'] = (mock_values['thermal_conductivity'] /
                                             (mock_values['density'] * mock_values['specific_heat']))

        return {prop: mock_values[prop] for prop in properties if prop in mock_values}

    def _get_default_property(self, prop: str, fluid: str) -> float:
        """Get default property value"""
        defaults = {
            'density': 1.225,
            'viscosity': 1.81e-5,
            'thermal_conductivity': 0.024,
            'specific_heat': 1005.0,
            'enthalpy': 300000.0,
            'entropy': 6900.0
        }
        return defaults.get(prop, 1.0)


class ChemicalReactionProperties:
    """Handle chemical reaction properties using Cantera"""

    def __init__(self):
        self.cantera_available = CANTERA_AVAILABLE
        self.mechanisms = {
            'methane_air': 'gri30.yaml',
            'hydrogen_air': 'h2o2.yaml',
            'simple_combustion': 'gri30.yaml'
        }

    def create_gas_mixture(self, mechanism: str = 'gri30.yaml',
                          composition: Optional[Dict[str, float]] = None) -> Optional[object]:
        """Create gas mixture using Cantera"""
        if not self.cantera_available:
            print("Cantera not available. Returning mock gas object.")
            return self._create_mock_gas(composition)

        try:
            if composition is None:
                composition = {'CH4': 1, 'O2': 2, 'N2': 7.52}  # Stoichiometric methane-air

            gas = ct.Solution(mechanism)
            gas.set_equivalence_ratio(1.0, composition)
            return gas

        except Exception as e:
            print(f"Error creating gas mixture: {e}")
            return self._create_mock_gas(composition)

    def _create_mock_gas(self, composition: Optional[Dict[str, float]] = None):
        """Create mock gas object for testing"""

        class MockGas:
            def __init__(self, comp):
                self.T = 300.0  # Temperature in K
                self.P = 101325.0  # Pressure in Pa
                self.composition = comp or {'CH4': 1, 'O2': 2, 'N2': 7.52}

            def set_TP(self, T, P):
                self.T = T
                self.P = P

            @property
            def density(self):
                return self.P / (287.0 * self.T)  # Ideal gas approximation

            @property
            def viscosity(self):
                return 1.81e-5 * (self.T / 273.15) ** 0.7

            @property
            def thermal_conductivity(self):
                return 0.024 * (self.T / 273.15) ** 0.8

            @property
            def cp(self):
                return 1005.0  # J/kg/K

            def equilibrate(self, method):
                print(f"Mock equilibration with method: {method}")

            def net_production_rates(self):
                return np.random.random(10) * 1e-6  # Mock reaction rates

        return MockGas(composition)

    def calculate_reaction_rates(self, gas, temperature: float,
                                 pressure: float) -> Dict[str, np.ndarray]:
        """Calculate chemical reaction rates"""
        if not self.cantera_available:
            return self._mock_reaction_rates()

        try:
            gas.TP = temperature, pressure
            gas.equilibrate('HP')  # Equilibrate at constant enthalpy and pressure

            rates = {
                'production_rates': gas.net_production_rates,
                'destruction_rates': -gas.net_production_rates,  # Simplified
                'heat_release_rate': np.sum(gas.net_production_rates * gas.partial_molar_enthalpies)
            }

            return rates

        except Exception as e:
            print(f"Error calculating reaction rates: {e}")
            return self._mock_reaction_rates()

    def _mock_reaction_rates(self) -> Dict[str, np.ndarray]:
        """Generate mock reaction rates"""
        n_species = 10
        return {
            'production_rates': np.random.random(n_species) * 1e-6,
            'destruction_rates': np.random.random(n_species) * 1e-6,
            'heat_release_rate': np.random.random() * 1e6
        }


class PINNsMaterialInterface:
    """Interface for providing material properties to PINNs"""

    def __init__(self):
        self.fluid_props = MaterialProperties()
        self.reaction_props = ChemicalReactionProperties()

    def get_properties_tensor(self, coordinates: np.ndarray,
                              fluid: str = 'air', base_temp: float = 300.0,
                              base_pressure: float = 101325.0) -> Dict[str, np.ndarray]:
        """
        Get material properties as arrays for PINNs training

        Args:
            coordinates: Array of spatial coordinates (N, 2 or 3)
            fluid: Fluid type
            base_temp: Base temperature
            base_pressure: Base pressure

        Returns:
            Dictionary of property arrays
        """
        n_points = coordinates.shape[0]

        # For this example, assume uniform properties
        props = self.fluid_props.get_fluid_properties(
            fluid, base_temp, base_pressure,
            ['density', 'viscosity', 'thermal_conductivity', 'specific_heat']
        )

        # Convert to arrays for each point
        property_arrays = {}
        for prop, value in props.items():
            property_arrays[prop] = np.full(n_points, value)

        return property_arrays

    def create_source_terms(self, coordinates: np.ndarray,
                            solution_values: np.ndarray,
                            reaction_type: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Create source terms for PDE (e.g., chemical reactions, heat sources)"""
        n_points = coordinates.shape[0]

        source_terms = {
            'mass_source': np.zeros(n_points),
            'momentum_source_x': np.zeros(n_points),
            'momentum_source_y': np.zeros(n_points),
            'energy_source': np.zeros(n_points)
        }

        if reaction_type == 'combustion':
            temperature = solution_values[:, -1] if solution_values.shape[1] > 1 else 300.0
            reaction_rate = np.exp(-(1000.0 / temperature)) * 1e6  # Arrhenius-like
            source_terms['energy_source'] = reaction_rate

        return source_terms


def create_property_database():
    """Create a database of common fluid properties"""
    materials = MaterialProperties()

    fluids = ['air', 'water', 'hydrogen', 'methane', 'co2']
    temperatures = np.linspace(273.15, 773.15, 10)  # 0°C to 500°C
    pressure = 101325.0  # Standard pressure

    database = {}

    for fluid in fluids:
        database[fluid] = {}
        for temp in temperatures:
            props = materials.get_fluid_properties(fluid, temp, pressure)
            database[fluid][f'{temp:.1f}K'] = props

    return database


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Calculate material properties')
    parser.add_argument('--fluid', default='air', help='Fluid type')
    parser.add_argument('--temp', type=float, default=300.0, help='Temperature in K')
    parser.add_argument('--pressure', type=float, default=101325.0, help='Pressure in Pa')
    args = parser.parse_args()

    materials = MaterialProperties()
    props = materials.get_fluid_properties(args.fluid, args.temp, args.pressure)
    print(f"Material properties for {args.fluid} at T={args.temp} K and P={args.pressure} Pa:")
    for key, value in props.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()