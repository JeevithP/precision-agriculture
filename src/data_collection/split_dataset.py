import os, shutil, random

def split_dataset(original_dir, base_dir, train_ratio=0.8):
    classes = os.listdir(original_dir)
    os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "val"), exist_ok=True)

    for cls in classes:
        cls_dir = os.path.join(original_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if not images:
            print(f"⚠️ Skipping {cls}, no valid images found.")
            continue

        random.shuffle(images)

        split_point = int(len(images) * train_ratio)
        train_imgs = images[:split_point]
        val_imgs = images[split_point:]

        train_cls_dir = os.path.join(base_dir, "train", cls)
        val_cls_dir = os.path.join(base_dir, "val", cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)

        for img in train_imgs:
            try:
                shutil.copy(os.path.join(cls_dir, img), os.path.join(train_cls_dir, img))
            except Exception as e:
                print(f"❌ Could not copy {img} from {cls}: {e}")

        for img in val_imgs:
            try:
                shutil.copy(os.path.join(cls_dir, img), os.path.join(val_cls_dir, img))
            except Exception as e:
                print(f"❌ Could not copy {img} from {cls}: {e}")

if __name__ == "__main__":
    original_dir = r"C:\Users\JEEVITH\OneDrive\Documents\precision-agriculture\PlantVillage"
    base_dir = r"C:\Users\JEEVITH\OneDrive\Documents\precision-agriculture\data\PlantVillage"
    split_dataset(original_dir, base_dir)
