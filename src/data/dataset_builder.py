from src.data.road_vqa_dataset import RoadVQADataset
from src.data.drivelm import DriveLMDataset

def build_dataset(cfg, tokenizer, image_processor, split="train", data_usage=1.0):
    dataset_name = cfg.dataset.name.lower()
    
    if dataset_name == "road_vqa":
        return RoadVQADataset(
            data_path=cfg.dataset.data_path,
            image_folder=cfg.dataset.image_folder,
            tokenizer=tokenizer,
            image_processor=image_processor,
            split=split,
            split_ratio=(0.8, 0.2, 0.0), # have val set as test set as well for local DriveLM evaluation/benchmark
            data_usage=data_usage
        )
        
    elif dataset_name == "drivelm":
        data_path = None
        if split == "val":
            data_path = cfg.dataset.val_data_path
        else:
            if cfg.dataset.train_data_path is not None:
                data_path = cfg.dataset.train_data_path
            else:
                data_path = cfg.dataset.data_path  # fallback to old single path if train_data_path is not specified
        
        return DriveLMDataset(
            data_path=data_path,
            image_folder=cfg.dataset.image_folder,
            tokenizer=tokenizer,
            image_processor=image_processor,
            split=split,
            # split_ratio=(0.8, 0.2, 0.0), # have val set as test set as well for local DriveLM evaluation/benchmark
            split_ratio=(0.0, 1.0, 0.0), # have val set as test set as well for local DriveLM evaluation/benchmark
            
            data_usage=data_usage
        )
        
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")