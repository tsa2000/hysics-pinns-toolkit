#!/usr/bin/env python3
"""
SALOME Mesh Generator

Simple geometry creation and mesh generation using SALOME platform.
Supports basic 2D/3D geometries with structured/unstructured meshing.

Dependencies: salome_meca, numpy
Usage: python salome_mesh_generator.py
"""

import numpy as np

try:
    import salome
    salome.salome_init()
    import GEOM
    from salome.geom import geomBuilder
    import SMESH
    from salome.smesh import smeshBuilder
except ImportError:
    print("Warning: SALOME not available. Running in mock mode.")

class SalomeMeshGenerator:
    def __init__(self):
        if 'salome' in globals():
            self.geom = geomBuilder.New()
            self.mesh = smeshBuilder.New()
        else:
            self.geom = None
            self.mesh = None

    def create_rectangle(self, width=1.0, height=1.0, name="Rectangle"):
        if not self.geom:
            print(f"Mock: Creating rectangle {width}x{height}")
            return None
        rect = self.geom.MakeFaceHW(width, height, 1)
        self.geom.addToStudy(rect, name)
        return rect

    def create_box(self, length=1.0, width=1.0, height=1.0, name="Box"):
        if not self.geom:
            print(f"Mock: Creating box {length}x{width}x{height}")
            return None
        box = self.geom.MakeBoxDXDYDZ(length, width, height)
        self.geom.addToStudy(box, name)
        return box

    def create_cylinder(self, radius=0.5, height=1.0, name="Cylinder"):
        if not self.geom:
            print(f"Mock: Creating cylinder r={radius}, h={height}")
            return None
        cylinder = self.geom.MakeCylinderRH(radius, height)
        self.geom.addToStudy(cylinder, name)
        return cylinder

    def generate_mesh(self, geometry, element_size=0.1, mesh_name="Mesh"):
        if not self.mesh:
            print(f"Mock: Generating mesh with element size {element_size}")
            return self._create_mock_mesh()
        mesh = self.mesh.Mesh(geometry)
        regular_1d = mesh.Segment()
        regular_1d.LocalLength(element_size)
        mefisto_2d = mesh.Triangle()
        mefisto_2d.MaxElementArea(element_size**2)
        try:
            netgen_3d = mesh.Tetrahedron()
            netgen_3d.MaxElementVolume(element_size**3)
        except:
            pass
        mesh.Compute()
        mesh.SetName(mesh_name)
        return mesh

    def _create_mock_mesh(self):
        x = np.linspace(0, 1, 11)
        y = np.linspace(0, 1, 11)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack([X.flatten(), Y.flatten(), np.zeros(len(X.flatten()))])
        return {
            'points': points,
            'cells': [],
            'name': 'MockMesh'
        }

    def export_mesh(self, mesh, filename, format_type="med"):
        if not self.mesh:
            print(f"Mock: Exporting mesh to {filename}.{format_type}")
            return True
        try:
            if format_type.lower() == "med":
                mesh.ExportMED(f"{filename}.med")
            elif format_type.lower() == "unv":
                mesh.ExportUNV(f"{filename}.unv")
            elif format_type.lower() == "stl":
                mesh.ExportSTL(f"{filename}.stl")
            else:
                print(f"Unsupported format: {format_type}")
                return False
            print(f"Mesh exported to {filename}.{format_type}")
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False

def main():
    generator = SalomeMeshGenerator()
    rect = generator.create_rectangle(2.0, 1.0, "FlowDomain")
    box = generator.create_box(1.0, 1.0, 1.0, "HeatDomain")
    cylinder = generator.create_cylinder(0.3, 1.0, "Pipe")
    geometries = [rect, box, cylinder]
    names = ["flow_mesh", "heat_mesh", "pipe_mesh"]
    for geom, name in zip(geometries, names):
        if geom is not None:
            mesh = generator.generate_mesh(geom, element_size=0.05, mesh_name=name)
            generator.export_mesh(mesh, name, "med")
    print("Mesh generation complete!")

if __name__ == "__main__":
    main()