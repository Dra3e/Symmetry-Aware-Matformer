from mp_api.client import MPRester


api_key = "exOmUg204h1fbQsVK65z9U5YLGXK453p" 
def fetch_materiale(material_id):
    with MPRester(api_key) as mpr:
        structure = mpr.materials.get_structure_by_material_id(material_id)
    return structure

# Example Usage
material_id = "mp-149"  # Example: Silicon
structure = fetch_materiale(material_id)
