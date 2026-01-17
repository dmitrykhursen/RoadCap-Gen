from src.data.road_vqa_dataset import RoadVQADataset
from src.data.drivelm import DriveLMDataset

def build_dataset(cfg, tokenizer, image_processor, split="train", data_usage=1.0):
    dataset_name = cfg.dataset.name.lower()
    
    if dataset_name == "roadvqa":
        return RoadVQADataset(
            data_path=cfg.dataset.data_path,
            image_folder=cfg.dataset.image_folder,
            tokenizer=tokenizer,
            image_processor=image_processor,
            split=split,
            data_usage=data_usage
        )
        
    elif dataset_name == "drivelm":
        return DriveLMDataset(
            data_path=cfg.dataset.data_path,
            image_folder=cfg.dataset.image_folder,
            tokenizer=tokenizer,
            image_processor=image_processor,
            split=split,
            data_usage=data_usage
        )
        
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")