from PIL import Image
import io

def raw_image_loader(path):
    with open(path, "rb") as f:
        return f.read()


def raw_image_folder_sample_to_pil_sample(xy):
    x, y = xy
    return Image.open(io.BytesIO(x)).convert("RGB"), y