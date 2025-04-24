import open3d as o3d
import numpy as np

def create_cylinder_mesh(center, radius, height, axis='Z', resolution=20):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, 
        height=height, 
        resolution=resolution
    )
    
    if axis == 'X':
        mesh.rotate([0, np.pi/2, 0], center=(0, 0, 0))
    elif axis == 'Y':
        mesh.rotate([np.pi/2, 0, 0], center=(0, 0, 0))
    
    mesh.translate(center)
    return mesh

def visualize_solids(solids):
    geometries = []
    
    for solid in solids:
        if solid['type'] == 'CYLINDER':
            mesh = create_cylinder_mesh(
                center=solid['center'],
                radius=solid['radius'],
                height=solid['height'],
                axis=solid['axis']
            )
            geometries.append(mesh)
        
        elif solid['type'] == 'COMPOSITE':
            base_mesh = create_cylinder_mesh(
                center=solid['solids'][0]['center'],
                radius=solid['solids'][0]['radius'],
                height=solid['solids'][0]['height'],
                axis=solid['solids'][0]['axis']
            )
            
            for sub_solid in solid['solids'][1:]:
                sub_mesh = create_cylinder_mesh(
                    center=sub_solid['center'],
                    radius=sub_solid['radius'],
                    height=sub_solid['height'],
                    axis=sub_solid['axis']
                )
                
                if solid['operation'] == 'UNION':
                    base_mesh = base_mesh + sub_mesh
                elif solid['operation'] == 'SUBTRACTION':
                    base_mesh = base_mesh - sub_mesh
            
            geometries.append(base_mesh)
    
    for mesh in geometries:
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    o3d.visualization.draw_geometries([coord_frame] + geometries)
