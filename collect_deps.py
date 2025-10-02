# collect_deps.py

import pickle
import ast
import sys
from tqdm import tqdm

def get_dependencies_from_item(item):
    """Extracts library names from a single data item using AST."""
    model_code = item.get('other_metadata', {}).get('code_prompt', '') + \
                 item.get('other_metadata', {}).get('canonical_solution', '')
    test_code = item.get("other_metadata", {}).get("test", "")
    full_code_for_ast = model_code + "\n" + test_code

    all_raw_imports = set()
    try:
        tree = ast.parse(full_code_for_ast)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    all_raw_imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    all_raw_imports.add(node.module.split('.')[0])
    except Exception:
        # Ignore items with syntax errors for dependency collection
        return set()
    return all_raw_imports

def main():
    # --- CONFIGURE THE FILE TO SCAN HERE ---
    file_to_scan = 'data/code/test_500.pkl'
    
    print(f"🔍 Scanning '{file_to_scan}' for all dependencies...")

    with open(file_to_scan, 'rb') as f:
        data_to_scan = pickle.load(f)

    master_dependency_set = set()
    for item in tqdm(data_to_scan, desc="Collecting dependencies"):
        deps = get_dependencies_from_item(item)
        master_dependency_set.update(deps)

    # --- Translate import names to pip package names ---
    translation_map = {
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'dateutil': 'python-dateutil',
        'skimage': 'scikit-image',
    }
    
    # Add known implicit dependencies
    if 'pandas' in master_dependency_set: master_dependency_set.add('openpyxl')
    if 'seaborn' in master_dependency_set: master_dependency_set.add('matplotlib')

    final_package_list = set()
    for lib in master_dependency_set:
        if lib:
            final_package_list.add(translation_map.get(lib, lib))

    # Remove standard libraries and non-installable modules
    non_installable = {'mpl_toolkits', 'builtins', None}
    std_libs = set(sys.stdlib_module_names)
    final_package_list -= non_installable
    final_package_list -= std_libs

    print("\n" + "="*50)
    print("✅ Dependency collection complete.")
    print("Run the following command in your activated conda environment:")
    print("="*50 + "\n")
    # Sort for consistent output
    sorted_packages = sorted(list(final_package_list))
    print(f"pip install {' '.join(sorted_packages)}")
    print("\n" + "="*50)

if __name__ == "__main__":
    main()