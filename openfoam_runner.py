#!/usr/bin/env python3

import os
import subprocess
import numpy as np
from pathlib import Path
import argparse

class OpenFOAMRunner:
    def __init__(self, case_dir="openfoam_case"):
        self.case_dir = Path(case_dir)
        self.foam_tutorials = os.getenv('FOAM_TUTORIALS', '/opt/openfoam/tutorials')

    def check_openfoam(self):
        try:
            result = subprocess.run(['which', 'icoFoam'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

    def create_cavity_case(self, size=1.0, viscosity=0.01, mesh_resolution=20):
        if not self.check_openfoam():
            print("OpenFOAM not found. Creating mock case structure.")
            return self._create_mock_case("cavity")

        self.case_dir.mkdir(parents=True, exist_ok=True)
        for d in ['0', 'constant', 'system']:
            (self.case_dir / d).mkdir(exist_ok=True)

        self._create_blockmesh_dict(size, mesh_resolution)
        self._create_velocity_field(viscosity)
        self._create_pressure_field()
        self._create_control_dict()
        self._create_fv_schemes()
        self._create_fv_solution()
        self._create_transport_properties(viscosity)

        print(f"Cavity case created in {self.case_dir}")
        return True

    def create_heat_conduction_case(self, size=1.0, thermal_diffusivity=1e-5):
        if not self.check_openfoam():
            print("OpenFOAM not found. Creating mock case structure.")
            return self._create_mock_case("heat")

        self.case_dir.mkdir(parents=True, exist_ok=True)
        for d in ['0', 'constant', 'system']:
            (self.case_dir / d).mkdir(exist_ok=True)

        self._create_temperature_field()

        print(f"Heat conduction case created in {self.case_dir}")
        return True

    def _create_mock_case(self, case_type):
        self.case_dir.mkdir(parents=True, exist_ok=True)
        if case_type == "cavity":
            self._generate_mock_cavity_data()
        elif case_type == "heat":
            self._generate_mock_heat_data()
        return True

    def _generate_mock_cavity_data(self):
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, y)
        U = np.sin(np.pi * X) * np.cos(np.pi * Y)
        V = -np.cos(np.pi * X) * np.sin(np.pi * Y)
        P = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        np.savez(self.case_dir / 'cavity_results.npz', x=x, y=y, X=X, Y=Y, U=U, V=V, P=P)
        print("Mock cavity data generated")

    def _generate_mock_heat_data(self):
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, y)
        T = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-2 * np.pi**2 * 0.1)
        np.savez(self.case_dir / 'heat_results.npz', x=x, y=y, X=X, Y=Y, T=T)
        print("Mock heat data generated")

    def _create_blockmesh_dict(self, size, resolution):
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

vertices
(
    (0 0 0)
    ({size} 0 0)
    ({size} {size} 0)
    (0 {size} 0)
    (0 0 0.1)
    ({size} 0 0.1)
    ({size} {size} 0.1)
    (0 {size} 0.1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({resolution} {resolution} 1) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    movingWall
    {{
        type wall;
        faces
        (
            (3 7 6 2)
        );
    }}
    fixedWalls
    {{
        type wall;
        faces
        (
            (0 4 7 3)
            (2 6 5 1)
            (1 5 4 0)
        );
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);

mergePatchPairs
(
);

// ************************************************************************* //
"""
        with open(self.case_dir / 'system' / 'blockMeshDict', 'w') as f:
            f.write(content)

    def _create_velocity_field(self, viscosity):
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    movingWall
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }
    fixedWalls
    {
        type            noSlip;
    }
    frontAndBack
    {
        type            empty;
    }
}

// ************************************************************************* //
"""
        with open(self.case_dir / '0' / 'U', 'w') as f:
            f.write(content)

    def _create_pressure_field(self):
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    movingWall
    {
        type            zeroGradient;
    }
    fixedWalls
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}

// ************************************************************************* //
"""
        with open(self.case_dir / '0' / 'p', 'w') as f:
            f.write(content)

    def _create_control_dict(self):
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     icoFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.5;
deltaT          0.005;
writeControl    timeStep;
writeInterval   20;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

// ************************************************************************* //
"""
        with open(self.case_dir / 'system' / 'controlDict', 'w') as f:
            f.write(content)

    def _create_fv_schemes(self):
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
    grad(p)         Gauss linear;
    grad(U)         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear orthogonal;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         orthogonal;
}

// ************************************************************************* //
"""
        with open(self.case_dir / 'system' / 'fvSchemes', 'w') as f:
            f.write(content)

    def _create_fv_solution(self):
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-06;
        relTol          0.05;
    }
    pFinal
    {
        $p;
        relTol          0;
    }
    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0;
    }
}

PISO
{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}

// ************************************************************************* //
"""
        with open(self.case_dir / 'system' / 'fvSolution', 'w') as f:
            f.write(content)

    def _create_transport_properties(self, viscosity):
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

nu              nu [0 2 -1 0 0 0 0] {viscosity};

// ************************************************************************* //
"""
        with open(self.case_dir / 'constant' / 'transportProperties', 'w') as f:
            f.write(content)

    def _create_temperature_field(self):
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      T;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 1 0 0 0];

internalField   uniform 0;

boundaryField
{
    defaultFaces
    {
        type            zeroGradient;
    }
    fixedWalls
    {
        type            fixedValue;
        value           uniform 300;
    }
    frontAndBack
    {
        type            empty;
    }
}

// ************************************************************************* //
"""
        with open(self.case_dir / '0' / 'T', 'w') as f:
            f.write(content)

def main():
    parser = argparse.ArgumentParser(description="OpenFOAM case setup and run")
    parser.add_argument("--case", type=str, default="cavity", choices=["cavity", "heat"], help="Type of case")
    parser.add_argument("--time", type=float, default=0.1, help="Simulation end time")
    parser.add_argument("--case_dir", type=str, default="openfoam_case", help="Case directory")
    args = parser.parse_args()

    runner = OpenFOAMRunner(case_dir=args.case_dir)

    if args.case == "cavity":
        runner.create_cavity_case()
    elif args.case == "heat":
        runner.create_heat_conduction_case()

if __name__ == "__main__":
    main()