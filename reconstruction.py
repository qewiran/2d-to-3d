import numpy as np

def find_rotation_axes(primitives_zy, primitives_xz, center_tolerance=10.0):
    axes_candidates = {'Z': [], 'X': []}

    # Анализ проекции XZ (ось Z)
    xz_circles = [p for p in primitives_xz if p['type'] == 'CIRCLE']
    xz_arcs = [p for p in primitives_xz if p['type'] == 'ARC']
    
    if xz_circles or xz_arcs:
        centers = [(p['center'][0], p['center'][1]) for p in xz_circles + xz_arcs]
        if centers:
            centers = np.array(centers)
            mean_center = np.mean(centers, axis=0)
            for p in xz_circles + xz_arcs:
                dist = np.sqrt((p['center'][0] - mean_center[0])**2 + 
                            (p['center'][1] - mean_center[1])**2)
                if dist < center_tolerance:
                    axes_candidates['Z'].append({
                        'center': p['center'],
                        'radius': p['radius']
                    })

    # Анализ проекции ZY (ось X)
    zy_circles = [p for p in primitives_zy if p['type'] == 'CIRCLE']
    zy_arcs = [p for p in primitives_zy if p['type'] == 'ARC']
    
    if zy_circles or zy_arcs:
        centers = [(p['center'][0], p['center'][1]) for p in zy_circles + zy_arcs]
        if centers:
            centers = np.array(centers)
            mean_center = np.mean(centers, axis=0)
            for p in zy_circles + zy_arcs:
                dist = np.sqrt((p['center'][0] - mean_center[0])**2 + 
                            (p['center'][1] - mean_center[1])**2)
                if dist < center_tolerance:
                    axes_candidates['X'].append({
                        'center': p['center'],
                        'radius': p['radius']
                    })

    return axes_candidates

def match_primitives(primitives_zy, primitives_xz, axes_candidates, radius_tolerance=10.0, coord_tolerance=10.0):
    matches = []

    for circle in axes_candidates['Z']:
        circle_center = circle['center']
        circle_radius = circle['radius']
        
        profile_primitives = []
        for prim in primitives_zy:
            if prim['type'] == 'LINE':
                x1, y1 = prim['start']
                x2, y2 = prim['end']
                dist1 = abs(x1 - circle_center[0])
                dist2 = abs(x2 - circle_center[0])
                if (abs(dist1 - circle_radius) < radius_tolerance or 
                    abs(dist2 - circle_radius) < radius_tolerance):
                    if abs(y1 - circle_center[1]) < coord_tolerance or \
                       abs(y2 - circle_center[1]) < coord_tolerance:
                        profile_primitives.append(prim)
            elif prim['type'] == 'ARC':
                arc_center_x = prim['center'][0]
                arc_radius = prim['radius']
                if abs(arc_center_x - circle_center[0]) < coord_tolerance and \
                   abs(arc_radius - circle_radius) < radius_tolerance:
                    profile_primitives.append(prim)
        
        if profile_primitives:
            matches.append({
                'axis': 'Z',
                'circle': circle,
                'profile': profile_primitives
            })

    for circle in axes_candidates['X']:
        circle_center = circle['center']
        circle_radius = circle['radius']
        
        profile_primitives = []
        for prim in primitives_xz:
            if prim['type'] == 'LINE':
                x1, y1 = prim['start']
                x2, y2 = prim['end']
                dist1 = abs(y1 - circle_center[1])
                dist2 = abs(y2 - circle_center[1])
                if (abs(dist1 - circle_radius) < radius_tolerance or 
                    abs(dist2 - circle_radius) < radius_tolerance):
                    if abs(x1 - circle_center[0]) < coord_tolerance or \
                       abs(x2 - circle_center[0]) < coord_tolerance:
                        profile_primitives.append(prim)
            elif prim['type'] == 'ARC':
                arc_center_y = prim['center'][1]
                arc_radius = prim['radius']
                if abs(arc_center_y - circle_center[1]) < coord_tolerance and \
                   abs(arc_radius - circle_radius) < radius_tolerance:
                    profile_primitives.append(prim)
        
        if profile_primitives:
            matches.append({
                'axis': 'X',
                'circle': circle,
                'profile': profile_primitives
            })

    return matches

def reconstruct_solids(matches):
    solids = []

    for match in matches:
        axis = match['axis']
        circle = match['circle']
        profile = match['profile']

        center = circle['center']
        radius = circle['radius']

        if not profile:
            continue

        if axis == 'Z':
            min_y = float('inf')
            max_y = float('-inf')
            for prim in profile:
                if prim['type'] == 'LINE':
                    y1, y2 = prim['start'][1], prim['end'][1]
                    min_y = min(min_y, y1, y2)
                    max_y = max(max_y, y1, y2)
                elif prim['type'] == 'ARC':
                    y_center = prim['center'][1]
                    r_arc = prim['radius']
                    min_y = min(min_y, y_center - r_arc)
                    max_y = max(max_y, y_center + r_arc)

            if min_y == float('inf') or max_y == float('-inf'):
                continue

            height = max_y - min_y
            solid = {
                'type': 'CYLINDER',
                'axis': 'Z',
                'center': (center[0], 0, min_y + height / 2),
                'radius': radius,
                'height': height
            }
            solids.append(solid)

        elif axis == 'X':
            min_x = float('inf')
            max_x = float('-inf')
            for prim in profile:
                if prim['type'] == 'LINE':
                    x1, x2 = prim['start'][0], prim['end'][0]
                    min_x = min(min_x, x1, x2)
                    max_x = max(max_x, x1, x2)
                elif prim['type'] == 'ARC':
                    x_center = prim['center'][0]
                    r_arc = prim['radius']
                    min_x = min(min_x, x_center - r_arc)
                    max_x = max(max_x, x_center + r_arc)

            if min_x == float('inf') or max_x == float('-inf'):
                continue

            height = max_x - min_x
            solid = {
                'type': 'CYLINDER',
                'axis': 'X',
                'center': (min_x + height / 2, center[1], 0),
                'radius': radius,
                'height': height
            }
            solids.append(solid)

    return solids

def handle_interactions(solids, overlap_tolerance=10.0):
    final_solids = []
    processed = set()

    for i, solid1 in enumerate(solids):
        if i in processed:
            continue

        solid1_axis = solid1['axis']
        solid1_center = solid1['center']
        solid1_radius = solid1['radius']
        solid1_height = solid1['height']

        if solid1_axis == 'Z':
            z_min = solid1_center[2] - solid1_height / 2
            z_max = solid1_center[2] + solid1_height / 2
            x_center, y_center = solid1_center[0], solid1_center[1]
        else:
            x_min = solid1_center[0] - solid1_height / 2
            x_max = solid1_center[0] + solid1_height / 2
            y_center, z_center = solid1_center[1], solid1_center[2]

        composite = {'type': 'COMPOSITE', 'operation': 'UNION', 'solids': [solid1]}
        for j, solid2 in enumerate(solids):
            if j == i or j in processed:
                continue

            solid2_axis = solid2['axis']
            solid2_center = solid2['center']
            solid2_radius = solid2['radius']
            solid2_height = solid2['height']

            intersect = False
            if solid1_axis == solid2_axis == 'Z':
                z_min2 = solid2_center[2] - solid2_height / 2
                z_max2 = solid2_center[2] + solid2_height / 2
                if (z_min < z_max2 + overlap_tolerance and 
                    z_max > z_min2 - overlap_tolerance):
                    dist = ((solid1_center[0] - solid2_center[0])**2 + 
                            (solid1_center[1] - solid2_center[1])**2)**0.5
                    if dist < solid1_radius + solid2_radius + overlap_tolerance:
                        intersect = True
            elif solid1_axis == solid2_axis == 'X':
                x_min2 = solid2_center[0] - solid2_height / 2
                x_max2 = solid2_center[0] + solid2_height / 2
                if (x_min < x_max2 + overlap_tolerance and 
                    x_max > x_min2 - overlap_tolerance):
                    dist = ((solid1_center[1] - solid2_center[1])**2 + 
                            (solid1_center[2] - solid2_center[2])**2)**0.5
                    if dist < solid1_radius + solid2_radius + overlap_tolerance:
                        intersect = True

            if intersect:
                if solid2_radius < solid1_radius:
                    composite['operation'] = 'SUBTRACTION'
                    composite['solids'].append(solid2)
                else:
                    composite['solids'].append(solid2)
                processed.add(j)

        if len(composite['solids']) > 1:
            final_solids.append(composite)
        else:
            final_solids.append(solid1)
        processed.add(i)

    return final_solids
