import argparse
import cv2
import torch
import csv
import numpy as np

from loguru import logger
from ultralytics import YOLO

from price_tag_recognition.vlm_inference import run_vlm_batch, initialize_vlm
from price_tag_recognition.utils.visualize import plot_tracking
from trackers.ocsort_tracker.ocsort import OCSort
from trackers.tracking_utils.timer import Timer


def crop_quality(img):
    if img is None or img.size == 0:
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    area = img.shape[0] * img.shape[1]

    return sharpness * 0.7 + area * 0.3


class YOLOPredictor:
    def __init__(self, model_path, device="cpu", conf=0.1, rotate=False):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.rotate = rotate

    def inference(self, frame, timer):
        height, width = frame.shape[:2]

        if self.rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        timer.tic()

        results = self.model.predict(
            frame,
            conf=self.conf,
            device=self.device,
            verbose=False
        )

        timer.toc()

        dets = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                dets.append([x1, y1, x2, y2, float(scores[i])])

        dets = np.array(dets) if len(dets) > 0 else None

        img_info = {
            "height": height,
            "width": width,
            "raw_img": frame
        }

        return [dets], img_info

def get_hw_after_rotation(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Could not read video")

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return frame.shape[0], frame.shape[1]

def imageflow_demo(predictor, args):

    cap = cv2.VideoCapture(args.video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if args.rotate:
        height, width = get_hw_after_rotation(args.video_path)

    if args.out_path:
        writer = cv2.VideoWriter(
            args.out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )
    else:
        writer = None

    tracker = OCSort(
        det_thresh=args.track_thresh,
        iou_threshold=args.iou_thresh,
        use_byte=args.use_byte
    )

    vlm_model, vlm_processor = initialize_vlm(device=args.device)

    best_crops = {}  # {id: (quality, crop)}

    timer = Timer()
    frame_id = 0

    logger.info("Starting imageflow...")

    while True:
        if frame_id == 100:
            break
            
        ret, frame = cap.read()
        if not ret:
            break

        outputs, img_info = predictor.inference(frame, timer)

        dets = outputs[0]

        if dets is not None and len(dets) > 0:
            dets = torch.from_numpy(dets).float()
            dets = dets.reshape(-1, 5)

            online_targets = tracker.update(
                dets,
                [img_info["height"], img_info["width"]],
                [img_info["height"], img_info["width"]]
            )
            
            online_tlwhs = []
            online_ids = []

            for t in online_targets:
                x1, y1, x2, y2, tid = t[:5]

                tlwh = [x1, y1, x2 - x1, y2 - y1]

                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

                    x1_, y1_, x2_, y2_ = map(int, [x1, y1, x2, y2])
                    crop = frame[y1_:y2_, x1_:x2_]
                    q = crop_quality(crop)
                    if tid not in best_crops or q > best_crops[tid][0]:
                        best_crops[tid] = (q, crop.copy())

            timer.toc()

            vis_frame = plot_tracking(
                img_info["raw_img"],
                online_tlwhs,
                online_ids,
                frame_id=frame_id + 1,
                fps=1.0 / max(timer.average_time, 1e-6)
            )

        else:
            vis_frame = img_info["raw_img"]

        if writer:
            writer.write(vis_frame)

        frame_id += 1

        if frame_id % 20 == 0:
            logger.info(f"Frame {frame_id}")

    cap.release()
    if writer:
        writer.release()

    ids = []
    images = []

    for tid, (_, crop) in best_crops.items():
        if crop is None:
            continue
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue

        ids.append(tid)
        images.append(crop)

    texts = run_vlm_batch(
        images,
        vlm_model,
        vlm_processor,
        batch_size=8
    )

    output_csv = args.csv_out_path or 'result.csv'

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text"])

        for tid, text in zip(ids, texts):
            writer.writerow([tid, text])


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = YOLOPredictor(
        model_path=args.ckpt,
        device=args.device,
        conf=args.conf,
        rotate=args.rotate
    )

    imageflow_demo(predictor, args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--csv_out_path", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--track_thresh", type=float, default=0.5)
    parser.add_argument("--iou_thresh", type=float, default=0.3)
    parser.add_argument("--min_box_area", type=float, default=10)
    parser.add_argument("--use_byte", action="store_true")
    parser.add_argument("--rotate", action="store_true")
    args = parser.parse_args()
    main(args)