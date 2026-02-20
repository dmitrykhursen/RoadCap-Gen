# Copyright (c) OpenMMLab. All rights reserved.
from ast import arg
import os
import sys
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
# from ultralytics import YOLO
import cv2

# yoloWorld
from mmdet.apis import init_detector
from mmyolo.registry import VISUALIZERS
from mmengine.dataset import Compose

# segment anything
from segment_anything import build_sam, SamPredictor, build_sam_vit_b, build_sam_vit_l

from mmdet.apis import init_detector, inference_detector as mmdet_inference_detector

# tracker
# 1. Calculate paths
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))) # Go up 3 levels to RoadCap-Gen
external_path = os.path.join(project_root, "external")

# 2. Add 'external' to sys.path
if external_path not in sys.path:
    sys.path.append(external_path)

# 3. Add 'project_root' to sys.path (for 'src' imports)
if project_root not in sys.path:
    sys.path.append(project_root)

# 4. Import using the library name directly (NOT external.fast_reid)
from tracker.mc_bot_sort import BoTSORT 
# OR if your mc_bot_sort is custom and lives in external/tracker:
# from external.tracker.mc_bot_sort import BoTSORT

# import colors
# from colormap import Colormap
from datetime import datetime

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

# from pone_dataset import PONELoader
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List
from src.data.nuscenes_dataset import NuscenesDataset
from src.data.frontcam_dataset import FRONTCAMDataset
import torchvision.transforms as T


def list_drive_lm_scenes(root_dir: str) -> List[str]:
    """
    Scans for leaf directories (scenes) that contain images.
    Assumes structure: root_dir / CAM_NAME / SCENE_ID / images...
    """
    root_path = Path(root_dir)
    scene_folders = []
    
    print(f"🔍 Scanning {root_dir} for scenes...")
    
    # Method 1: Robust walk (finds any folder containing .jpg/.png)
    # This is safer if directory depth varies
    for root, dirs, files in os.walk(root_path):
        # If this folder contains images, treat it as a scene
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
            scene_folders.append(str(Path(root)))

    # Method 2: Strict structure (Faster if structure is guaranteed)
    # root_path = Path(root_dir)
    # for cam_dir in root_path.iterdir():
    #     if cam_dir.is_dir():
    #         for scene_dir in cam_dir.iterdir():
    #             if scene_dir.is_dir():
    #                 scene_folders.append(str(scene_dir))

    return sorted(scene_folders)  # Sort is CRITICAL for deterministic chunking


def list_all_zip_files(root_dir: str) -> List[str]:
    # Lists all CAMERA zip files in the given directory and its subdirectories.
    root_path = Path(root_dir)
    zip_files = [str(p) for p in root_path.rglob("*.zip") if "camera" in str(p).lower()]
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

def get_output_dir_from_nuscenes_folder(input_folder_path, args_out_dir, input_root):
    """
    Creates output path mirroring relative structure.
    Input: /data/nuscenes/CAM_FRONT/n008-2018...
    Root:  /data/nuscenes/
    Output: /output/CAM_FRONT/n008-2018...
    """
    input_path = Path(input_folder_path).resolve()
    root_path = Path(input_root).resolve()
    out_root = Path(args_out_dir).resolve()

    try:
        # Get relative path (e.g., CAM_FRONT/n008...)
        relative_path = input_path.relative_to(root_path)
    except ValueError:
        # Fallback if path manipulation fails
        relative_path = input_path.name

    return out_root / relative_path

def inference_detector(model, inputs, score_thr, nms_thr):
    with torch.no_grad():
        _output = model.test_step(inputs)
        output = []
        for _out in _output:
            pred_instances = _out.pred_instances
            keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
            pred_instances = pred_instances[keep_idxs]
            pred_instances = _out.pred_instances[_out.pred_instances.scores.float() > score_thr]
            _out.pred_instances = pred_instances
            output.append(_out)
    return output

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--img', help='Image path, include image file, dir and URL.')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--det-model', choices=["yolov11", "yoloworld"], help='Detection model to use "yolov11" or "yoloworld"')
    parser.add_argument('--det-model-config', default=None, help='Path to detection model config file (only for yoloworld)')
    parser.add_argument('--open-classes', help='Open classes list for YoloWorld')

    parser.add_argument('--dataset', help='Choose dataset type to process either "valeo" or "drivelm"')
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

    if args.dataset == "valeo":
        print(f"Processing Valeo dataset")
    elif args.dataset == "drivelm":
        print(f"Processing DriveLM dataset")
    else:
        raise ValueError("Choose dataset to process: either valeo or drivelm") 

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
    
    
    # color_list = colormap()
    # We create a list of list: [[R,G,B], [R,G,B], ...]
    color_list = np.random.randint(0, 255, size=(200, 3)).tolist()
    
    # build the detection model
    model = None
    if args.det_model == "yolov11":
        print("Using YOLOv11 detection model")
        model = YOLO(args.checkpoint)
    elif args.det_model == "yoloworld":
        print("Using YOLOworld detection model")
        
        if args.open_classes.endswith('.txt'):
            with open(args.open_classes) as f:
                lines = f.readlines()
            open_classes = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
        else:
            open_classes = [[t.strip()] for t in args.open_classes.split(',')]
        
        # init YOLO World
        model = init_detector(args.det_model_config, args.checkpoint, device=args.device)
        model.cfg.visualizer['line_width'] = 10
        # visualizer = VISUALIZERS.build(model.cfg.visualizer)
        model.dataset_meta['classes'] = tuple([l[0] for l in open_classes])
        # visualizer.dataset_meta = model.dataset_meta
        # visualizer.mask_color = [matplotlib.colormaps['tab20'](i)[:3] for i in range(len(model.dataset_meta['classes']))]

        # build test pipeline
        # model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        # reparameterize texts / open classes for YOLOworld
        model.reparameterize(open_classes)
        # the dataset_meta is loaded from the checkpoint and
        # then pass to the model in init_detector
        model.dataset_meta['classes'] = tuple([l[0] for l in open_classes])
    else:
        raise ValueError("Choose detection model: either yolov11 or yoloworld")

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
        T.ToTensor(),
        T.Lambda(lambda img: T.functional.pad(img, (0, (32 - img.shape[1] % 32) % 32, 32 - (img.shape[2] % 32), 0))),  # Pad to have H and W divisible by 32
        T.Lambda(lambda x: x.permute(0, 1, 2)),
    ])


    # --- 1. GET ALL SCENE FOLDERS ---
    if os.path.isdir(args.img):
        if args.dataset == "drivelm":
            all_scene_folders = list_drive_lm_scenes(args.img)
        elif args.dataset == "valeo":
            all_scene_folders = list_all_zip_files(args.img)
    else:
        # Single folder case
        all_scene_folders = [args.img]
    
    total_scenes = len(all_scene_folders)

    # # just take CAM_FRONT scenes (quick for debug)
    # all_scene_folders = [p for p in all_scene_folders if "CAM_FRONT" in p]

    print(f"Found {total_scenes} total scenes/folders.")
    print(all_scene_folders)
    print(" - - - - - "*3)

    for iz, scene_path in enumerate(all_scene_folders):
        # init tracker
        tracker = BoTSORT(args, frame_rate=0.0)
        # NuScenes camera captures at 12Hz
        # tracker = BoTSORT(args, frame_rate=12.0)
        # tracker = BoTSORT(args, frame_rate=30.0) # but actualy the delta is not the same and it's the difference 1-5 seconds...



        # Get current date and time as a string, e.g. "2025-06-10_14-30-45"
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # init saving result format, result for the whole video of the scene of a sensor type
        results = {
            # "name": "2D_segmentation",
            "name": args.det_model + ("_SAM" if args.use_sam else "") + ("_SimpleTrack" if args.use_tracking else ""),
            "time_stamp_generation": ''.join(str(time.time()).split('.')),
            "date_time": date_time,
            "img_path": args.img,
            "model": args.det_model,
            "model_version": "1",
            # "sensor": folder_2.replace(folder_1+'_', '').replace(".npz.zip",""),
            "output_type": "BB+SAM_masks+tracking" if args.use_sam else "BB",
            "results": []
        }

        if args.dataset == "valeo":
            output_dir = get_output_dir(scene_path, args.out_dir)
            # output_dir = get_output_dir_from_nuscenes_folder(scene_path, args.out_dir, args.img)#args.out_dir
        elif args.dataset == "drivelm":
            output_dir = get_output_dir_from_nuscenes_folder(scene_path, args.out_dir, args.img)

        print(f"{output_dir=}")

        # 1. Initialize Dataset First
        dataset = None
        if args.dataset == "valeo":
            dataset = FRONTCAMDataset(image_folder=scene_path, transform=transform) 
            # dataset = PONELoader(zip_file_path=zip_path, transform=transform, file_extension="jpg", sample_per_zipfile=1)
        elif args.dataset == "drivelm":
            if args.det_model == "yoloworld":
                transform = T.Compose([
                T.ToTensor()
            ])
                
            dataset = NuscenesDataset(image_folder=scene_path, transform=transform)


        # 2. Get Input Count from Dataset instead of os.listdir
        input_count = len(dataset)
        
        # 3. Decision Logic (Check if the folder already processed)
        is_complete = False
        output_count = 0
        print(f"Checking output directory: {output_dir}, exists: {output_dir.exists()}, glob(*.json): {len(list(output_dir.glob('*.json'))) if output_dir.exists() else 'None'}")
        if output_dir.exists():
            if args.show:
                output_count = len(list((output_dir / "vis").glob("*.png")))
            else:
                output_count = len(list(output_dir.glob("*.json")))
                is_complete = True # if just folder exists, than it was processed
            
            if output_count >= input_count and not args.overwrite:
                is_complete = True

        if is_complete:
            print(f"⏩ [{iz+1}/{total_scenes}] Skipping: {output_dir} (processed {output_count} from {input_count}) \n - - - - - "*3)
            continue
        else:
            print(f"🚀 [{iz+1}/{total_scenes}] Processing: {output_dir} (processed {output_count} from {input_count})")

        # 4. Process Dataloader
        dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=False, collate_fn=safe_collate)

        for data in tqdm(dataloader):
            if data is None or data.get('img') is None:
                continue  # Skip broken sample

            results["results"] = []  # empty results for the next frames
            images, filepaths = data['img'], data['filepath']

            if args.det_model == "yolov11":
                det_results = model(images, imgsz=1280, conf=args.score_thr)
            elif args.det_model == "yoloworld":
                
                # # 3. Run inference
                # det_results = inference_detector(model,
                #                         inputs=mmdet_data,
                #                         score_thr=args.score_thr,
                #                         nms_thr=args.nms_thr)
                from mmengine.dataset import Compose, pseudo_collate
                
                # 1. Grab YOLO-World's official preprocessor pipeline
                test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

                # 2. Build the dictionaries and run them through the pipeline
                data_list = []
                for filepath in filepaths:
                    data_info = {
                        'img_path': str(filepath),
                        'img_id': 0,
                        'texts': open_classes  # <-- We successfully inject the texts here!
                    }
                    # test_pipeline natively reads the image, pads it to 32-stride, and builds DetDataSample
                    data_list.append(test_pipeline(data_info))
                
                # 3. Collate the results into the exact batch format MMDetection wants
                mmdet_data = pseudo_collate(data_list)
                
                # 4. Use YOUR original custom inference function (the one at line 204)
                det_results = inference_detector(model,
                                        inputs=mmdet_data,
                                        score_thr=args.score_thr,
                                        nms_thr=args.nms_thr)
                # print(det_results)
                

            # go frame by frame for tracking and sam
            for img, det_res, filepath in zip(images, det_results, filepaths):
                filename = os.path.basename(filepath)

                # init frame result format
                frame_results = {"image_name": filename,
                                "time_stamp": filename.split('_')[-1].split('.')[0],
                                "yolo_feat": []
                                } # to be appended in results['results']

                # Get candidate predict info based on the model type
                if args.det_model == "yolov11":
                    pred_instances = det_res
                    bboxes = pred_instances.boxes.xyxy  # [N, 4]
                    scores = pred_instances.boxes.conf  # [N]
                    labels = pred_instances.boxes.cls   # [N]
                    dataset_classes = pred_instances.names
                elif args.det_model == "yoloworld":
                    pred_instances = det_res.pred_instances
                    bboxes = pred_instances.bboxes      # [N, 4]
                    scores = pred_instances.scores      # [N]
                    labels = pred_instances.labels      # [N]
                    dataset_classes = model.dataset_meta['classes']

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

                    img = (img * 255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

                                # cv2.rectangle(vis_image,(int(box_numpy[0,0]),int(box_numpy[0,1])),(int(box_numpy[0,2]),int(box_numpy[0,3])),class_palette[int(tcls)],2)
                                cv2.rectangle(vis_image,(int(box_numpy[0,0]),int(box_numpy[0,1])),(int(box_numpy[0,2]),int(box_numpy[0,3])),class_palette[int(tcls) % 10], 2)
                            else:
                                # xyxy to x_tl, y_tl, w, h
                                box_copy = box.copy()
                                box_copy[2] -= box_copy[0]
                                box_copy[3] -= box_copy[1]
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

                                box_counter += 1
                                frame_results["yolo_feat"].append(object_pred)

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
    import time
    import random
    # for long karolina runs to not have the same folder processed
    time.sleep(random.uniform(1, 17))

    main()
    print("FINISHED.")

# salloc -A eu-25-10 -p qgpu_exp --gpus-per-node 1 -t 1:00:00 --nodes 1
# salloc -A open-36-7 -p qgpu_exp --gpus-per-node 1 -t 1:00:00 --nodes 1
# source /mnt/proj1/eu-25-10/envs/yolov11_sam_tracking/bin/activate
# cd /mnt/proj1/eu-25-10/dmytro/RoadCap-Gen

# python scripts/00_data_prep/extract_metadata.py --img /mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/nuscenes/train_val_samples_grouped/ --checkpoint external/models/YOLOv11/best.pt --segment-ckpt external/models/SAM/sam_vit_h_4b8939.pth --use-tracking --bs=1 --out-dir="output/drivelm_yolo/" --dataset="drivelm" --fast-reid-config external/fast_reid/configs/MOT17/sbs_S50.yml --fast-reid-weights external/models/Tracker/mot17_sbs_S50.pth

# nuscenes datat to process using yolov11
# python scripts/00_data_prep/extract_metadata.py --img /scratch/project/eu-25-10/datasets/nuScenes/samples_grouped --checkpoint external/models/YOLOv11/best.pt --segment-ckpt external/models/SAM/sam_vit_h_4b8939.pth --use-tracking --bs=1 --out-dir="/mnt/proj1/eu-25-10/datasets/nuScenes_metadata/tmp_yolo" --dataset="drivelm" --fast-reid-config external/fast_reid/configs/MOT17/sbs_S50.yml --fast-reid-weights external/models/Tracker/mot17_sbs_S50.pth

# nuscenes datat to process using yoloWorld
# source /mnt/proj1/eu-25-10/envs/yoloworld-seg-track/bin/activate
# python scripts/00_data_prep/extract_metadata.py --img /scratch/project/eu-25-10/datasets/nuScenes/samples_grouped --checkpoint external/models/YoloWorld/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth --det-model-config external/models/YoloWorld/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py --open-classes external/models/YoloWorld/open_classes_nuscnesgt_minus_yolov11_short_names.txt --det-model yoloworld  --segment-ckpt external/models/SAM/sam_vit_h_4b8939.pth --use-tracking --bs=1 --out-dir="/mnt/proj1/eu-25-10/datasets/nuScenes_metadata/tmp_yoloWorld" --dataset="drivelm" --fast-reid-config external/fast_reid/configs/MOT17/sbs_S50.yml --fast-reid-weights external/models/Tracker/mot17_sbs_S50.pth

# valeo data to process using yolov11
# python scripts/00_data_prep/extract_metadata.py --img /scratch/project/eu-25-10/datasets/FRONT_CAM_zipped/20250527/18 --checkpoint external/models/YOLOv11/best.pt --segment-ckpt external/models/SAM/sam_vit_h_4b8939.pth --use-tracking --bs=1 --out-dir=/scratch/project/eu-25-10/datasets/FRONT_CAM_zipped_metadata/YOLOv11_1/ --dataset="valeo" --fast-reid-config external/fast_reid/configs/MOT17/sbs_S50.yml --fast-reid-weights external/models/Tracker/mot17_sbs_S50.pth