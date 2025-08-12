import json

# Define file paths with absolute paths
input_txt_path = 'D:/AIffel/EfficientViT/efficientvit/new_classes.txt'
output_json_path = 'D:/AIffel/EfficientViT/data/mapillary-vistas-dataset_public_v2.0/custom_class_mapping.json'

# Define the new class order
new_class_names = [
    'sidewalk', 'curb-cut', 'crosswalk', 'road', 'minor-road', 'bike-lane', 'curb', 'terrain',
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'rider', 'vegetation', 'sky', 'water',
    'sign', 'traffic-cone', 'puddle', 'pothole', 'manhole', 'catch-basin', 'bench', 'pole',
    'building', 'wall', 'fence', 'void'
]

# Create a mapping from new class name to new ID (0-29 for main classes, 255 for void)
new_name_to_id = {name: i for i, name in enumerate(new_class_names) if name != 'void'}
new_name_to_id['void'] = 255

class_mapping = {}

with open(input_txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or ':' not in line:
            continue
        
        new_name, old_ids_str = line.split(':', 1)
        new_name = new_name.strip()
        
        if new_name not in new_name_to_id:
            continue
            
        new_id = new_name_to_id[new_name]
        
        old_ids_str = old_ids_str.strip()
        if old_ids_str == '없음':
            continue
            
        # Handle complex string like '115116' -> '115', '116'
        if '115116' in old_ids_str:
            old_ids_str = old_ids_str.replace('115116', '115, 116, ')

        old_ids = [int(x.strip()) for x in old_ids_str.split(',') if x.strip()]
        
        for old_id in old_ids:
            class_mapping[old_id] = new_id

with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(class_mapping, f, indent=4)

print(f"Successfully created mapping file: {output_json_path}")