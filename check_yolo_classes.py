from ultralytics import YOLO

model = YOLO('yolov8l.pt')
print("Total YOLO classes:", len(model.names))
print("\nAll classes:")
for i, name in model.names.items():
    print(f"{i}: {name}")

# Check for iron-related
iron_keywords = ['iron', 'hair', 'dryer', 'appliance', 'electric']
print("\n\nIron/Hair dryer related:")
for i, name in model.names.items():
    if any(keyword in name.lower() for keyword in iron_keywords):
        print(f"{i}: {name}")

