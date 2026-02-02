# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser
import numpy as np

from tqdm import tqdm

from ultralytics import YOLO
import cv2

# segment anything
from segment_anything import build_sam, SamPredictor, build_sam_vit_b, build_sam_vit_l

# tracker
from tracker.mc_bot_sort import BoTSORT

# import colors
from colormap import colormap
from datetime import datetime

# image processing
from lfccalib.calibration_2d import demosaicing, undistort_image

# for tracking
import torch
import json
from torchvision.ops import nms

# mask encoding
import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib
import io
from PIL import Image
import time

from pone_dataset import PONELoader
from torch.utils.data import DataLoader
# from mmengine.dataset import default_collate
import torchvision.transforms as T
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List

def list_all_zip_files(root_dir: str) -> List[str]:
    # Lists all zip files in the given directory and its subdirectories.
    root_path = Path(root_dir)
    zip_files = [str(p) for p in root_path.rglob("*.zip")]
    return zip_files

def encode_binary_mask(mask: np.ndarray):
  """Converts a numpy array binary mask into base64 encoded text."""

  # mask shape is [h,w]
  # check input mask --
  if mask.dtype != bool:
    raise ValueError(
        "encode_binary_mask expects a binary mask, received dtype == %s" %
        mask.dtype)

  mask = np.squeeze(mask)
  if len(mask.shape) != 2:
    raise ValueError(
        "encode_binary_mask expects a 2d mask, received shape == %s" %
        mask.shape)

  # convert input mask to expected COCO API input --
  mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
  mask_to_encode = mask_to_encode.astype(np.uint8)
  mask_to_encode = np.asfortranarray(mask_to_encode)

  # RLE encode mask --
  encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

  # compress and base64 encoding --
  binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str).decode('utf-8')
  return base64_str

def decode_binary_mask(rleCodedStr, imHeight, imWidth):
    uncodedStr = base64.b64decode(rleCodedStr)
    uncompressedStr = zlib.decompress(uncodedStr,wbits = zlib.MAX_WBITS)   
    detection ={
        'size': [imHeight, imWidth],
        'counts': uncompressedStr
    }
    detlist = []
    detlist.append(detection)
    mask = coco_mask.decode(detlist)
    binaryMask = mask.astype('bool') 
    return binaryMask[:,:,0]


def get_overlapped_img(img, masks, color):
    img_mask = img.copy()
    for mask in masks.copy():
        mask = np.stack([mask, mask, mask], axis=-1)
        mask = mask.astype(np.uint8)*255
        # Convert the white color (for blobs) to magenta
        mask_colored = mask
        mask_colored[mask_colored[:,:,0] == 255, 0] = color[0]
        mask_colored[mask_colored[:,:,1] == 255, 1] = color[1]
        mask_colored[mask_colored[:,:,2] == 255, 2] = color[2]

        # print("mask_colored shape:", mask_colored.shape)

        img_mask = cv2.addWeighted(img_mask,0.4,np.array(mask_colored),0.3,0)
    
    img_mask[masks.astype(np.float32).sum(0)==0] = img[masks.astype(np.float32).sum(0)==0]
    return img_mask

def get_output_dir(zip_path, args_out_dir):
    # === 1. Resolve paths ===
    zip_path = Path(zip_path).resolve()
    out_root = Path(args_out_dir).resolve()

    # === 2. Get relative path from PONE_zipped ===
    zip_parts = zip_path.parts
    if "PONE_zipped" in zip_parts:
        idx = zip_parts.index("PONE_zipped")
        relative_path_after_pone = Path(*zip_parts[idx + 1:])
        output_dir = out_root / relative_path_after_pone.with_suffix('')  # drop .zip
    elif "FRONT_CAM_zipped" in zip_parts:
        idx = zip_parts.index("FRONT_CAM_zipped")
        relative_path_after = Path(*zip_parts[idx + 1:])
        output_dir = out_root / relative_path_after.with_suffix('')  # drop .zip
    else:
        print(f'Using default output structure, zip: {zip_path}')
        output_dir = out_root / "out"

    return output_dir


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--img', help='Image path, include image file, dir and URL.')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--bs', type=int, default=1, help='Batch size')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--overwrite', action='store_true', help='Overwrite the results.')

    parser.add_argument(
        '--segment-ckpt', default='./model_zoo/sam_vit_h_4b8939.pth', help='Path to segment anything model.')

    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    # parser.add_argument(
    #     '--deploy',
    #     action='store_true',
    #     help='Switch model to deployment mode')
    parser.add_argument(
        '--score-thr', type=float, default=0.2, help='Bbox score threshold')
    parser.add_argument(
        '--class-name',
        nargs='+',
        type=str,
        help='Only Save those classes if set')
    parser.add_argument(
        '--to-labelme',
        action='store_true',
        help='Output labelme style label file')
    parser.add_argument('--nms-thr', default=0.5, type=float, help='NMS threshold.')
    

    parser.add_argument(
        '--fc', action='store_true', help='front cam, need demosaic and undo distortion.')
    # OLD version of front cam!!!

    # SAM
    parser.add_argument(
        '--use-sam', action='store_true', help='Use SAM segmentator.')
    
    # tracking args
    parser.add_argument('--use-tracking', action='store_true', help='Use (object) Simple Tracker')
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.2, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.3, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=4, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"./fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"./pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.1,
                        help='threshold for rejecting low appearance similarity reid matches')
    parser.add_argument('--name', default='exp', help='save results to project/name')

    # multi thread 
    parser.add_argument(
        '--chunk_idx', type=int, default=0, help='batch_number')
    
    parser.add_argument(
        '--num_chunks', type=int, default=1, help='number of chunks')
    #args.ablation = False
    
    args = parser.parse_args()
    args.with_reid = True
    args.ablation = False
    return args

def safe_collate(batch):
    # To catch None in the batch
    batch = [item for item in batch if item is not None]
    if not batch:
        return None  # skip batch entirely if it's all bad
    return torch.utils.data.default_collate(batch)

def main():
    args = parse_args()

    # load segment anything model, by default, I load the biggest one, vit-h.
    # change it by giving different path to segment-ckpt
    segmenter = None
    if args.use_sam:
        print("Loading pretrained SAM: ", args.segment_ckpt)
        if "vit_h" in args.segment_ckpt:
            segmenter = SamPredictor(build_sam(checkpoint=args.segment_ckpt))
        elif "vit_l" in args.segment_ckpt:
            segmenter = SamPredictor(build_sam_vit_l(checkpoint=args.segment_ckpt))
        elif "vit_b" in args.segment_ckpt:
            segmenter = SamPredictor(build_sam_vit_b(checkpoint=args.segment_ckpt))
    
    
    color_list = colormap()
    
    # build the YOLOv8 model from a config file and a checkpoint file
    model = YOLO(args.checkpoint)

    if args.use_sam:
        segmenter.model.to(args.device)
    # print("segmenter.device ", segmenter.device)

    class_palette = {
    0: (255, 0, 0),     # Red
    1: (0, 255, 0),     # Green
    2: (0, 0, 255),     # Blue
    3: (255, 255, 0),   # Yellow
    4: (0, 255, 255),   # Cyan
    5: (255, 0, 255),   # Magenta
    6: (128, 0, 128),   # Purple
    7: (255, 165, 0),   # Orange
    8: (0, 128, 128),   # Teal
    9: (128, 128, 0)    # Olive
    }

    transform = T.Compose([
        # T.Resize(640*2, interpolation=T.InterpolationMode.BILINEAR),
        # T.CenterCrop(640*2),  # Ensures the final size is exactly 640x640
        T.ToTensor(),
        T.Lambda(lambda img: T.functional.pad(img, (0, (32 - img.shape[1] % 32) % 32, 32 - (img.shape[2] % 32), 0))),  # Pad to have H and W divisible by 32
        T.Lambda(lambda x: x.permute(0, 1, 2)),
    ])

    if os.path.isdir(args.img):
        # process a directory of zip files
        zip_files = list_all_zip_files(args.img)
    else:
        # process a single zip file
        zip_files = [args.img]
    
    print(f"zip files: {zip_files}")

    for iz, zip_path in enumerate(zip_files):
        # init tracker
        tracker = BoTSORT(args, frame_rate=0.0)

        # Get current date and time as a string, e.g. "2025-06-10_14-30-45"
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # init saving result format, result for the whole video of the scene of a sensor type
        results = {
            # "name": "2D_segmentation",
            "name": "YOLOv11" + ("_SAM" if args.use_sam else "") + ("_SimpleTrack" if args.use_tracking else ""),
            "time_stamp_generation": ''.join(str(time.time()).split('.')),
            "date_time": date_time,
            "img_path": args.img,
            "model": "YOLOv11",
            "model_version": "1",
            # "sensor": folder_2.replace(folder_1+'_', '').replace(".npz.zip",""),
            "output_type": "BB",
            "results": []
        }

        output_dir = get_output_dir(zip_path, args.out_dir)
        print(f"{output_dir=}")
        # if output folder exists then this folder was processed already before, so skip it
        if output_dir.exists() and not args.overwrite:
            json_files = list(output_dir.glob("*.json"))
            if json_files:
                print(f"+ + + [{iz+1}/{len(zip_files)}] Output folder already exists, args.overwrite is False and JSON files found, so skip it: {output_dir} + + +")
                continue
#        elif "CAM_SUR_WM_LE_WE" in zip_path or "CAM_SUR_WM_RI_EA" in zip_path:
#            # (PONE_zipped dataset)
#            print(
#                f"+ + + [{iz + 1}/{len(zip_files)}] Right or Left SUR RI/LE (under mirror camera) DON'T process yet, so skip it: {zip_path} + + +")
#            continue
        elif "fcs.zip" in zip_path or "gnss.zip" in zip_path or "livox.zip" in zip_path or "camera1.zip" in zip_path:
            # don't process these zip files (FRONT_CAM_zipped dataset)
           print(
               f"+ + + [{iz + 1}/{len(zip_files)}] FCS/GNSS/LIVOX.zip don't process, so skip it: {zip_path} + + +")
           continue
        else:
            print(f"= = = = = [{iz+1}/{len(zip_files)}] Processing zip file: {zip_path} = = = = = =")

        dataset = PONELoader(zip_file_path=zip_path, transform=transform, file_extension="jpg", sample_per_zipfile=1)
        dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=False, collate_fn=safe_collate)
        # print(f"dataloader len: {len(dataloader)}")

        for data in tqdm(dataloader):
            if data is None or data.get('img') is None:
                continue  # Skip broken sample

            results["results"] = []  # empty results for the next frames
            images, filepaths = data['img'], data['filepath']

            det_results = model(images, imgsz=1280, conf=args.score_thr)
            # print(f"{det_results}=")

            # go frame by frame for tracking and sam
            for img, det_res, filepath in zip(images, det_results, filepaths):
                filename = os.path.basename(filepath)

                # init frame result format
                frame_results = {"image_name": filename,
                                "time_stamp": filename.split('_')[-1].split('.')[0],
                                "yolo_feat": []
                                } # to be appended in results['results']

                # Get candidate predict info with score threshold
                pred_instances = det_res
                bboxes = pred_instances.boxes.xyxy  # [N, 4]
                scores = pred_instances.boxes.conf  # [N] xyxy
                labels = pred_instances.boxes.cls  # [N]
                dataset_classes = pred_instances.names
                # Apply NMS
                keep_idxs = nms(bboxes, scores, iou_threshold=args.nms_thr)
                # Filter predictions
                bboxes, scores, labels = bboxes[keep_idxs], scores[keep_idxs], labels[keep_idxs]

                if args.use_tracking:
                    # tracking
                    dets = []
                    if bboxes.shape[0]>0:
                        dets = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1).float(), torch.zeros_like(bboxes)], dim=-1)
                        dets = dets.cpu().numpy()

                    # Assuming image is a tensor of shape (1, 3, H, W)
                    if isinstance(img, torch.Tensor):
                        # Convert from (1, 3, H, W) -> (H, W, 3)
                        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()

                    # img = cv2.convertScaleAbs(img)
                    img = (img * 255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # print(f"{img=}")
                    # print(f"{dets=}")

                    online_targets = tracker.update(dets, img) # tracking results for this frame

                    # give the img to segmenter backbone
                    if args.use_sam:
                        segmenter.set_image(img)

                    vis_image = img.copy()
                    box_counter = 0 # unique within the frame
                    for t in online_targets: # for loop objects in this frame
                        tlwh = t.tlwh
                        box = t.tlbr
                        tid = t.track_id
                        tcls = t.cls
                        clsname = dataset_classes[int(tcls)]
                        tscore = t.score
                        #print(tcls)
                        if tlwh[2] * tlwh[3] > args.min_box_area:
                            # predict masks using segmenter #
                            if args.use_sam:
                                masks, mask_quality, _ = segmenter.predict(box=box)

                                # we observe that segmenter prediction masks for the most salient object in the box #
                                masks = masks.sum(0)
                                # print("masks.shape ", masks.shape)
                                mask_compressed = encode_binary_mask(masks.astype(bool).copy())
                                plot_masks = masks.copy()[np.newaxis,:,:].astype(np.bool_)


                            if args.show: # if plot
                                # plot class name and confidence score
                                display_text = f"{clsname}: {tscore:.2f}" # tscore is rounded to 2 decimals

                                # Calculate text size
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.25
                                thickness = 1
                                text_size = cv2.getTextSize(display_text, font, font_scale, thickness)[0]

                                # Define text position (inside the top-left of the bounding box)
                                text_x, text_y = int(box[0]), int(box[1]) + text_size[1] + 5  # Inside top-left corner

                                # Draw black background for text inside the box
                                padding_x = 1  # Horizontal padding
                                padding_y = 1  # Vertical padding

                                # print(f"{vis_image=}")
                                # print(f"{vis_image.shape=}")

                                cv2.rectangle(
                                    vis_image,
                                    (text_x - padding_x, text_y - text_size[1] - padding_y),  # Adjust top-left corner
                                    (text_x + text_size[0] + padding_x, text_y + padding_y),  # Adjust bottom-right corner
                                    (0, 0, 0),  # Black background
                                    -1  # Filled rectangle
                                )

                                # Draw white text inside the box
                                cv2.putText(
                                    vis_image, display_text,
                                    (text_x + 5, text_y),
                                    font, font_scale,
                                    (255, 255, 255),  # White text
                                    thickness, cv2.LINE_AA
                                )

                                # plot tracking
                                vis_image = cv2.putText(vis_image, f"{int(tid)}", (int(0.5*box[0]+0.5*box[2]), int(0.5*box[1]+0.5*box[3])), cv2.FONT_HERSHEY_SIMPLEX, 2, (int(color_list[int(tid)%79][0]), int(color_list[int(tid)%79][1]), int(color_list[int(tid)%79][2])), 2, cv2.LINE_AA)

                                # plot masks
                                if args.use_sam:
                                    vis_image = get_overlapped_img(vis_image, plot_masks, color=color_list[int(tid)%79])

                                # plot box #
                                box_numpy = np.array(box)[np.newaxis, :]

                                vis_image = np.ascontiguousarray(vis_image)

                                cv2.rectangle(vis_image,(int(box_numpy[0,0]),int(box_numpy[0,1])),(int(box_numpy[0,2]),int(box_numpy[0,3])),class_palette[int(tcls)],2)
                            else:
                                # xyxy to x_tl, y_tl, w, h
                                # print("box before", box)
                                box_copy = box.copy()
                                box_copy[2] -= box_copy[0]
                                box_copy[3] -= box_copy[1]
                                # print("box: ", box)
                                # print("box_copy: ", box_copy)
                                # img_id = filename
                                object_pred = {
                                                'bbox': box_copy.tolist(),
                                                "bbox_id": box_counter,
                                                "object_id": tid,
                                                "score": round(float(tscore), 2),
                                                "category_name": clsname
                                }
                                if args.use_sam:
                                    object_pred.update({
                                        'mask': mask_compressed,
                                        'mask_h': masks.shape[0],
                                        'mask_w': masks.shape[1]
                                    })

                                #print("object_pred ", object_pred)
                                box_counter += 1
                                frame_results["yolo_feat"].append(object_pred)

                    #print("frame_results: ", frame_results)
                    results['results'].append(frame_results)
                else:
                    # only using Yolo detector
                    frame_results["yolo_feat"] = [
                        {
                            "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                            "bbox_id": i,
                            "score": round(float(scores[i]), 2),
                            "category_name": dataset_classes[int(labels[i])]
                        }
                        for i, box in enumerate(bboxes)
                    ]
                    results["results"].append(frame_results)

                # save this frame
                output_dir.mkdir(parents=True, exist_ok=True)
                filepath = Path(filepath).resolve()
                frame_stem = filepath.stem
                
                if args.show:
                    out_file_path = output_dir / "vis" / f"{frame_stem}.png"
                    out_file_path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Saving visualization to: {out_file_path}")
                    cv2.imwrite(str(out_file_path), vis_image)
                else:
                    # save results in json
                    out_file_path = output_dir / f"{frame_stem}.json"
                    out_file_path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Saving JSON to: {out_file_path}")
                    with open(out_file_path, 'w') as f:
                        json.dump(results, f, indent=4)
                        

if __name__ == '__main__':
    main()
    print("THE END.")

# TODO: Warning: not enough matching points

# cd valeo-projects/Detectors/yolov11-sam-tracking-main/
# source yolov11_sam_tracking/bin/activate
# python image_demo_withseg_tracking_with_zip_new_format.py --img /home/dmytro/valeo-projects/Detectors/wp5_yolo-world_sam/data/Veh01_20231026_102121_005_CAM_WFV_FW_CE_NO_ENC.npz.zip --checkpoint ./YOLOv11/best.pt --segment-ckpt ./sam/sam_vit_h_4b8939.pth --use-sam --use-tracking --show

# /mnt/proj1/eu-25-10/envs/yolov11_sam_tracking/bin/activate
# cd /scratch/project/eu-25-10/code/dmytro/valeo-projects
# /scratch/project/eu-25-10/datasets/PONE_zipped/20241115/2/
# /scratch/project/eu-25-10/datasets/PONE_zipped_metadata/

# sbatch ./pipelines/yolov11_track_run.ch
#/media/dmytro/dkhursenUSB/work/exa4mind/debug/Veh01_20240320_054522_059_CAM_WFV_FW_CE_NO_ENC.zip
# /media/dmytro/FC_disk_X/Dmytro/FRONT_CAM_zipped/6

# python image_demo_withseg_tracking_with_zip_new_format.py --img /media/dmytro/dkhursenUSB/work/exa4mind/debug/frontcam5_20240410_114725/camera2.zip --checkpoint ./YOLOv11/best.pt --segment-ckpt ./sam/sam_vit_h_4b8939.pth --use-tracking