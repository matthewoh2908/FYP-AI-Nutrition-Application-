# reports how many training images/boxes exist per class after conversion to spot badly under-represented food classes before spending time training.
import argparse
import yaml
from pathlib import Path
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    with open(args.output_root / "data.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    names = config["names"]

    box_counts = Counter()
    image_counts = Counter()

    for split in ("train", "val"):
        label_dir = args.output_root / "labels" / split
        for label_file in label_dir.glob("*.txt"):
            classes_in_this_image = set()
            with open(label_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    class_id = int(line.split()[0])
                    box_counts[class_id] += 1
                    classes_in_this_image.add(class_id)
            for c in classes_in_this_image:
                image_counts[c] += 1

    print(f"{'Class':<25} {'Images':>8} {'Boxes':>8}")
    for class_id in sorted(names.keys()):
        name = names[class_id]
        print(f"{name:<25} {image_counts.get(class_id, 0):>8} {box_counts.get(class_id, 0):>8}")

    counts = [image_counts.get(c, 0) for c in names.keys()]
    if counts:
        print(f"\nMin images in a class: {min(counts)}")
        print(f"Max images in a class: {max(counts)}")
        print(f"Median images per class: {sorted(counts)[len(counts)//2]}")
        low = [names[c] for c in names if image_counts.get(c, 0) < 20]
        if low:
            print(f"\n{len(low)} classes have fewer than 20 training images "
                  f"and may be unreliable to evaluate or train well:")
            print(", ".join(low))


if __name__ == "__main__":
    main()