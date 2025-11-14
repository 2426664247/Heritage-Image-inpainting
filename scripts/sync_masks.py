import argparse
import os
from PIL import Image, ImageOps

def is_image_file(name):
    n = name.lower()
    return n.endswith((".png", ".jpg", ".jpeg"))

def list_image_relpaths(root):
    rels = []
    if not root:
        return rels
    for r, _, files in os.walk(root):
        for f in files:
            if is_image_file(f):
                full = os.path.join(r, f)
                rel = os.path.relpath(full, root)
                rels.append(rel)
    return rels

def ensure_parent_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def binarize_mask(img):
    img = img.convert("L")
    return img.point(lambda p: 255 if p >= 128 else 0)

def generate_inverse(src_dir, dst_dir, relpath, overwrite=False):
    src_path = os.path.join(src_dir, relpath)
    dst_path = os.path.join(dst_dir, relpath)
    if (not overwrite) and os.path.exists(dst_path):
        return False, dst_path
    try:
        img = Image.open(src_path)
    except Exception:
        return False, dst_path
    mask = binarize_mask(img)
    inv = ImageOps.invert(mask)
    ensure_parent_dir(dst_path)
    inv.save(dst_path)
    return True, dst_path

def sync_dirs(masks_dir, masks_inv_dir, overwrite=False, dry_run=False):
    created_to_masks = 0
    created_to_inverted = 0
    rels_masks = list_image_relpaths(masks_dir)
    rels_inverted = list_image_relpaths(masks_inv_dir)
    for rel in rels_masks:
        if dry_run:
            pass
        else:
            ok, _ = generate_inverse(masks_dir, masks_inv_dir, rel, overwrite)
            if ok:
                created_to_inverted += 1
    for rel in rels_inverted:
        if dry_run:
            pass
        else:
            ok, _ = generate_inverse(masks_inv_dir, masks_dir, rel, overwrite)
            if ok:
                created_to_masks += 1
    print(f"masks目录: {len(rels_masks)} 个文件")
    print(f"masks_inverted目录: {len(rels_inverted)} 个文件")
    if dry_run:
        print("干跑模式：不进行写入")
    else:
        print(f"生成到masks的反向掩码: {created_to_masks} 个")
        print(f"生成到masks_inverted的反向掩码: {created_to_inverted} 个")

def main():
    p = argparse.ArgumentParser(description="同步两个掩码文件夹并生成反向掩码")
    p.add_argument("--masks", type=str, default="masks")
    p.add_argument("--masks_inverted", type=str, default="masks_inverted")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    os.makedirs(args.masks, exist_ok=True)
    os.makedirs(args.masks_inverted, exist_ok=True)
    sync_dirs(args.masks, args.masks_inverted, overwrite=args.overwrite, dry_run=args.dry_run)

if __name__ == "__main__":
    main()