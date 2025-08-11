#!/usr/bin/env python3
import numpy as np
import meshio
import argparse
from pathlib import Path
from typing import Dict, Optional

class MeshioConverter:
    def __init__(self):
        self.supported_formats = {
            'input': ['.med', '.msh', '.vtk', '.vtu', '.unv', '.inp', '.stl'],
            'output': ['.vtk', '.vtu', '.msh', '.stl', '.ply', '.off']
        }

    def read_mesh(self, filename: str) -> Optional[meshio.Mesh]:
        try:
            mesh = meshio.read(filename)
            print(f"Successfully read mesh: {filename}")
            print(f"Points: {len(mesh.points)}")
            print(f"Cell types: {list(mesh.cells_dict.keys())}")
            return mesh
        except Exception as e:
            print(f"Error reading mesh {filename}: {e}")
            return None

    def write_mesh(self, mesh: meshio.Mesh, filename: str) -> bool:
        try:
            meshio.write(filename, mesh)
            print(f"Successfully wrote mesh: {filename}")
            return True
        except Exception as e:
            print(f"Error writing mesh {filename}: {e}")
            return False

    def convert_mesh(self, input_file: str, output_file: str) -> bool:
        mesh = self.read_mesh(input_file)
        if mesh is None:
            return False
        return self.write_mesh(mesh, output_file)

    def extract_points(self, mesh: meshio.Mesh) -> np.ndarray:
        return mesh.points

    def extract_cells(self, mesh: meshio.Mesh, cell_type: str = "triangle") -> np.ndarray:
        if cell_type in mesh.cells_dict:
            return mesh.cells_dict[cell_type]
        else:
            available = list(mesh.cells_dict.keys())
            print(f"Cell type '{cell_type}' not found. Available: {available}")
            return np.array([])

    def extract_boundary_points(self, mesh: meshio.Mesh) -> Dict[str, np.ndarray]:
        points = mesh.points
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        tolerance = 1e-10
        boundaries = {
            'left': points[np.abs(points[:, 0] - x_min) < tolerance],
            'right': points[np.abs(points[:, 0] - x_max) < tolerance],
            'bottom': points[np.abs(points[:, 1] - y_min) < tolerance],
            'top': points[np.abs(points[:, 1] - y_max) < tolerance]
        }
        if points.shape[1] > 2:
            z_min, z_max = points[:, 2].min(), points[:, 2].max()
            boundaries.update({
                'front': points[np.abs(points[:, 2] - z_min) < tolerance],
                'back': points[np.abs(points[:, 2] - z_max) < tolerance]
            })
        return boundaries

    def create_training_points(self, mesh: meshio.Mesh, n_interior: int = 1000, 
                              n_boundary: int = 200) -> Dict[str, np.ndarray]:
        points = mesh.points
        bounds = {
            'x': [points[:, 0].min(), points[:, 0].max()],
            'y': [points[:, 1].min(), points[:, 1].max()]
        }
        if points.shape[1] > 2:
            bounds['z'] = [points[:, 2].min(), points[:, 2].max()]

        if points.shape[1] == 2:
            x_int = np.random.uniform(bounds['x'][0], bounds['x'][1], n_interior)
            y_int = np.random.uniform(bounds['y'][0], bounds['y'][1], n_interior)
            interior_points = np.column_stack([x_int, y_int])
        else:
            x_int = np.random.uniform(bounds['x'][0], bounds['x'][1], n_interior)
            y_int = np.random.uniform(bounds['y'][0], bounds['y'][1], n_interior)
            z_int = np.random.uniform(bounds['z'][0], bounds['z'][1], n_interior)
            interior_points = np.column_stack([x_int, y_int, z_int])

        boundary_points = self._generate_boundary_points(bounds, n_boundary, points.shape[1])
        return {
            'interior': interior_points,
            'boundary': boundary_points,
            'mesh_points': points
        }

    def _generate_boundary_points(self, bounds: Dict, n_points: int, dim: int) -> np.ndarray:
        boundary_points = []
        if dim == 2:
            n_per_side = n_points // 4
            x_bound = np.random.uniform(bounds['x'][0], bounds['x'][1], n_per_side)
            bottom = np.column_stack([x_bound, np.full(n_per_side, bounds['y'][0])])
            top = np.column_stack([x_bound, np.full(n_per_side, bounds['y'][1])])
            y_bound = np.random.uniform(bounds['y'][0], bounds['y'][1], n_per_side)
            left = np.column_stack([np.full(n_per_side, bounds['x'][0]), y_bound])
            right = np.column_stack([np.full(n_per_side, bounds['x'][1]), y_bound])
            boundary_points = np.vstack([bottom, top, left, right])
        else:
            n_per_face = n_points // 6
            for axis in range(3):
                for side in [0, 1]:
                    coords = []
                    for i in range(3):
                        if i == axis:
                            coords.append(np.full(n_per_face, bounds[list(bounds.keys())[i]][side]))
                        else:
                            bound_key = list(bounds.keys())[i]
                            coords.append(np.random.uniform(bounds[bound_key][0], bounds[bound_key][1], n_per_face))
                    boundary_points.append(np.column_stack(coords))
            boundary_points = np.vstack(boundary_points)
        return boundary_points

    def mesh_info(self, filename: str):
        mesh = self.read_mesh(filename)
        if mesh is None:
            return
        print(f"\n=== Mesh Information: {filename} ===")
        print(f"Number of points: {len(mesh.points)}")
        print(f"Spatial dimension: {mesh.points.shape[1]}")
        print(f"Point bounds:")
        for i, axis in enumerate(['x', 'y', 'z'][:mesh.points.shape[1]]):
            print(f"  {axis}: [{mesh.points[:, i].min():.3f}, {mesh.points[:, i].max():.3f}]")
        print(f"Cell types and counts:")
        for cell_type, cells in mesh.cells_dict.items():
            print(f"  {cell_type}: {len(cells)}")
        if mesh.point_data:
            print(f"Point data fields: {list(mesh.point_data.keys())}")
        if mesh.cell_data:
            print(f"Cell data fields: {list(mesh.cell_data.keys())}")

def main():
    parser = argparse.ArgumentParser(description='Convert mesh formats and extract PINNs data')
    parser.add_argument('input', help='Input mesh file')
    parser.add_argument('-o', '--output', help='Output mesh file')
    parser.add_argument('--info', action='store_true', help='Show mesh information')
    parser.add_argument('--extract-points', action='store_true', help='Extract training points')

    args = parser.parse_args()

    converter = MeshioConverter()

    if args.info:
        converter.mesh_info(args.input)

    if args.output:
        success = converter.convert_mesh(args.input, args.output)
        if success:
            print(f"Conversion completed: {args.input} -> {args.output}")

    if args.extract_points:
        mesh = converter.read_mesh(args.input)
        if mesh:
            training_data = converter.create_training_points(mesh)
            base_name = Path(args.input).stem
            np.savez(f"{base_name}_training_points.npz", **training_data)
            print(f"Training points saved to {base_name}_training_points.npz")

if __name__ == "__main__":
    main()