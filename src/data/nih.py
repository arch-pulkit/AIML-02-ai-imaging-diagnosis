import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NIH_DIR = PROJECT_ROOT/"data"/"raw"/"nih"

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
             "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
             "Pleural_Thickening", "Pneumonia", "Pneumothorax"]

def load_labels(csv_path) -> pd.DataFrame:

    df = pd.read_csv(csv_path)
    #drop
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    rename_dict ={
        "Image Index": "image_index",
        "Finding Labels": "finding_labels",
        "Follow-up #": "follow_up",
        "Patient ID": "patient_id",
        "Patient Age": "patient_age",
        "Patient Gender": "patient_sex",
        "View Position": "view_position",
        "OriginalImage[Width": "width",
            "Height]": "height",
        "OriginalImagePixelSpacing[x": "spacing_x",
            "y]": "spacing_y",
    }

    df = df.rename(columns=rename_dict, errors="raise")
    
    return df



def find_images(root) -> dict[str, Path]:
    image_dict = {}

    for p in Path(root).rglob("*.png"):
        image_dict[p.name] = p

    return image_dict