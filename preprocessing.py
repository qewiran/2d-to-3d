import cv2
import numpy as np

def create_primitives_list(circles, arcs, lines):
    primitives = []
    
    for x, y, r in circles:
        primitives.append({
            "type": "CIRCLE",
            "center": (x, y),
            "radius": r
        })
    
    for x, y, r, start, end in arcs:
        primitives.append({
            "type": "ARC",
            "center": (x, y),
            "radius": r,
            "start_angle": start,
            "end_angle": end
        })
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line)
            primitives.append({
                "type": "LINE",
                "start": (x1, y1),
                "end": (x2, y2)
            })
    
    return primitives

def preprocess_and_extract_proj(image_path, output_prefix="proj"):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    img_blur = cv2.GaussianBlur(img, (7, 7), sigmaX=2.0)
    _, img_bin = cv2.threshold(img_blur, 200, 255, cv2.THRESH_BINARY_INV)
   
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    results = []
    padding = 10
    for i, contour in enumerate(contours[:2]):
        x, y, w, h = cv2.boundingRect(contour)
        
        if w <= 0 or h <= 0:
            continue
            
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, img.shape[1])
        y2 = min(y + h + padding, img.shape[0])
        
        if y1 >= y2 or x1 >= x2:
            continue
            
        cropped = img_bin[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
            
        output_path = f"{output_prefix}{i}.png"
        cv2.imwrite(output_path, cropped)
        results.append(output_path)
        
    return results

def find_primitives_on_proj(image_path, center_tolerance=5.0, num_points=360, area_threshold=50.0):
    img_bin = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_bin is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    if len(np.unique(img_bin)) > 2:
        _, img_bin = cv2.threshold(img_bin, 200, 255, cv2.THRESH_BINARY_INV)

    cnts, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    potential_circles = []
    filtered_contours = []
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area > area_threshold:
            filtered_contours.append(c)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) > 5:
                (x, y), r = cv2.minEnclosingCircle(c)
                potential_circles.append((x, y, r))

    if potential_circles:
        centers = np.array([(x, y) for x, y, _ in potential_circles])
        mean_center = np.mean(centers, axis=0)
        valid_circles = [(x, y, r) for x, y, r in potential_circles 
                        if np.sqrt((x - mean_center[0])**2 + (y - mean_center[1])**2) < center_tolerance]
    else:
        valid_circles = []
    
    full_circles = []
    arcs_list = []
    for x, y, r in valid_circles:
        angles = np.linspace(0, 360, num_points, endpoint=False)
        points = [(int(x + r * np.cos(np.radians(angle))), int(y + r * np.sin(np.radians(angle))), angle) for angle in angles]
        points = [(px, py, angle) for px, py, angle in points 
                  if 0 <= px < img_bin.shape[1] and 0 <= py < img_bin.shape[0]]
        
        pixel_values = []
        for px, py, angle in points:
            patch = img_bin[max(0, py-2):min(img_bin.shape[0], py+3), 
                           max(0, px-2):min(img_bin.shape[1], px+3)]
            is_zero = np.all(patch == 0)
            pixel_values.append((angle, is_zero))
        
        arcs = []
        current_start = None
        has_gaps = False
        for i, (angle, is_zero) in enumerate(pixel_values):
            if not is_zero:
                if current_start is None:
                    current_start = angle
            else:
                has_gaps = True
                if current_start is not None:
                    prev_angle = pixel_values[i-1][0] if i > 0 else pixel_values[-1][0]
                    arcs.append((current_start, prev_angle))
                    current_start = None
        
        if current_start is not None and has_gaps:
            prev_angle = pixel_values[-1][0]
            arcs.append((current_start, prev_angle))
        
        if not has_gaps and current_start is not None:
            full_circles.append((x, y, r))
        elif arcs: 
            arcs_list.extend([(x, y, r, start, end) for start, end in arcs])
    
    unique_radii = set()
    filtered_full_circles = []
    for x, y, r in full_circles:
        is_unique = True
        for existing_r in unique_radii:
            if abs(r - existing_r) <= 3.0:
                is_unique = False
                break
        if is_unique:
            unique_radii.add(r)
            filtered_full_circles.append((x, y, r))
    full_circles = filtered_full_circles

    mask = np.zeros_like(img_bin)
    for x, y, r in full_circles:
        cv2.circle(mask, (int(x), int(y)), int(r), 255, 10)
    for x, y, r, start, end in arcs_list:
        cv2.ellipse(mask, (int(x), int(y)), (int(r), int(r)), 0, start, end, 255, 10)

    img_contours = np.zeros_like(img_bin)
    cv2.drawContours(img_contours, filtered_contours, -1, 255, 1)
    img_clean = cv2.bitwise_and(img_contours, cv2.bitwise_not(mask))
    img_dilated = cv2.dilate(img_clean, np.ones((3, 3), np.uint8), iterations=1)
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(img_dilated)[0]
    lines = [line[0] for line in lines] if lines is not None else []

    return full_circles, arcs_list, lines
