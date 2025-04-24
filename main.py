from preprocessing import preprocess_and_extract_proj, find_primitives_on_proj, create_primitives_list
from reconstruction import find_rotation_axes, match_primitives, reconstruct_solids, handle_interactions
from visualization import visualize_solids

def main():
    drawing_path = "bearing.png"  # Укажите путь к вашему чертежу
    
    # 1. Препроцессинг
    proj_paths = preprocess_and_extract_proj(drawing_path, "proj")
    zy_path = proj_paths[1]
    xz_path = proj_paths[0]
    
    # 2. Поиск примитивов
    circles_zy, arcs_zy, lines_zy = find_primitives_on_proj(zy_path)
    circles_xz, arcs_xz, lines_xz = find_primitives_on_proj(xz_path)
    
    zy_primitives = create_primitives_list(circles_zy, arcs_zy, lines_zy)
    xz_primitives = create_primitives_list(circles_xz, arcs_xz, lines_xz)
    
    # 3. Реконструкция
    axes = find_rotation_axes(zy_primitives, xz_primitives)
    matches = match_primitives(zy_primitives, xz_primitives, axes)
    solids = reconstruct_solids(matches)
    final_solids = handle_interactions(solids)
    
    # 4. Визуализация
    visualize_solids(final_solids)

if __name__ == "__main__":
    main()
