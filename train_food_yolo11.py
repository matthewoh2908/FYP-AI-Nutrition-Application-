# Trains YOLO11s on a food dataset, starting from COCO-pretrained weights
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11s on a food dataset.")
    parser.add_argument("--data", required=True, )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default="runs/food_yolo11")
    parser.add_argument("--name", default="uecfood256_yolo11s")
    args = parser.parse_args()

    # Start from YOLO11's COCO-pretrained weights
    model = YOLO("yolo11s.pt")

    # Train the model on the specified dataset
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )

    print("Validation mAP50-95:", results.box.map)
    print("Validation mAP50:", results.box.map50)


if __name__ == "__main__":
    main()
