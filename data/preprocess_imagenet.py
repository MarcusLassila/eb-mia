from io import BytesIO
from pathlib import Path
import argparse
import json
import warnings
import zipfile

from datasets import Image as DatasetImage
from datasets import load_dataset
from PIL import Image

def default_output_dir(data_dir, size):
    '''
    Build the default processed ImageNet cache directory.
    Args:
        data_dir (str | Path): Dataset storage root.
        size (int): Output image size.
    Returns:
        Path: Processed cache directory.
    '''
    return Path(data_dir) / "processed" / "ImageNet" / f"rgb-sz{size}"

def image_extension(image_format):
    '''
    Return the file extension for a PIL image format.
    Args:
        image_format (str): PIL image format.
    Returns:
        str: Filename extension.
    '''
    if image_format == "JPEG":
        return "jpg"
    return image_format.lower()

def open_raw_image(image_payload):
    '''
    Open a HuggingFace decode=False image payload without EXIF transpose.
    Args:
        image_payload (dict): Image payload with bytes or path.
    Returns:
        PIL.Image.Image: Open image object.
    '''
    image_bytes = image_payload["bytes"]
    image_path = image_payload["path"]
    if image_bytes is not None:
        image_buffer = BytesIO(image_bytes)
        return Image.open(image_buffer)
    return Image.open(image_path)

def preprocess_image(image_payload, size):
    '''
    Convert one raw image payload to clean fixed-size RGB.
    Args:
        image_payload (dict): HuggingFace decode=False image payload.
        size (int): Output image size.
    Returns:
        PIL.Image.Image: RGB resized image without source metadata.
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Metadata Warning.*",
            category=UserWarning,
            module="PIL.TiffImagePlugin",
        )
        warnings.filterwarnings(
            "ignore",
            message="Corrupt EXIF data.*",
            category=UserWarning,
            module="PIL.TiffImagePlugin",
        )
        with open_raw_image(image_payload) as image:
            rgb_image = image.convert("RGB")
            output_size = (size, size)
            resized_image = rgb_image.resize(output_size, resample=Image.Resampling.BICUBIC)
    return resized_image

def encode_image(image, image_format, quality):
    '''
    Encode a processed image without source metadata.
    Args:
        image (PIL.Image.Image): Processed RGB image.
        image_format (str): PIL image format.
        quality (int): JPEG quality.
    Returns:
        bytes: Encoded image bytes.
    '''
    image_buffer = BytesIO()
    if image_format == "JPEG":
        image.save(image_buffer, format=image_format, quality=quality)
    else:
        image.save(image_buffer, format=image_format)
    return image_buffer.getvalue()

def shard_relative_path(args, shard_index):
    '''
    Return the relative path for a shard.
    Args:
        args (argparse.Namespace): CLI arguments.
        shard_index (int): Shard index.
    Returns:
        str: Relative shard path.
    '''
    file_name = f"imagenet-rgb-sz{args.size}-{shard_index:06d}.zip"
    return str(Path("shards") / file_name)

def preprocess_split(split_name, data_dir, output_dir, start_index, args):
    '''
    Preprocess one ImageNet split in deterministic dataset order.
    Args:
        split_name (str): HuggingFace ImageNet split name.
        data_dir (Path): HuggingFace dataset cache directory.
        output_dir (Path): Processed cache directory.
        start_index (int): Global start index for deterministic ordering.
        args (argparse.Namespace): CLI arguments.
    Returns:
        tuple[int, int]: Number of processed images and next global index.
    '''
    dataset = load_dataset(
        "ILSVRC/imagenet-1k",
        split=split_name,
        cache_dir=str(data_dir),
        token=True,
    )
    dataset = dataset.cast_column("image", DatasetImage(decode=False))
    extension = image_extension(args.image_format)
    zip_file = None
    current_shard = None
    n_split_images = 0
    try:
        for split_index, sample in enumerate(dataset):
            if args.limit_per_split is not None and split_index >= args.limit_per_split:
                break
            global_index = start_index + split_index
            shard_index = global_index // args.images_per_shard
            shard_path = shard_relative_path(args, shard_index)
            if current_shard != shard_path:
                if zip_file is not None:
                    zip_file.close()
                full_shard_path = output_dir / shard_path
                full_shard_path.parent.mkdir(parents=True, exist_ok=True)
                zip_file = zipfile.ZipFile(full_shard_path, "w", compression=zipfile.ZIP_STORED)
                current_shard = shard_path
            member_name = f"{global_index:09d}.{extension}"
            image = preprocess_image(sample["image"], args.size)
            image_bytes = encode_image(image, args.image_format, args.quality)
            zip_file.writestr(member_name, image_bytes)
            n_split_images += 1
            n_processed = global_index + 1
            if args.log_every > 0 and n_processed % args.log_every == 0:
                print(f"processed {n_processed} images", flush=True)
    finally:
        if zip_file is not None:
            zip_file.close()
    next_index = start_index + n_split_images
    return n_split_images, next_index

def write_manifest(output_dir, args, split_lengths):
    '''
    Write the processed ImageNet manifest atomically.
    Args:
        output_dir (Path): Processed cache directory.
        args (argparse.Namespace): CLI arguments.
        split_lengths (dict): Number of cached images per split.
    Returns:
        Path: Written manifest path.
    '''
    manifest = {
        "version": 1,
        "dataset": "ImageNet",
        "size": args.size,
        "color": "RGB",
        "image_format": args.image_format,
        "images_per_shard": args.images_per_shard,
        "splits": [
            {"name": "train", "length": split_lengths["train"]},
            {"name": "validation", "length": split_lengths["validation"]},
        ],
    }
    manifest_path = output_dir / "manifest.json"
    temporary_path = output_dir / "manifest.json.tmp"
    with temporary_path.open("w") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    temporary_path.replace(manifest_path)
    return manifest_path

def build_parser():
    '''
    Build the ImageNet preprocessing CLI parser.
    Returns:
        argparse.ArgumentParser: Command-line parser.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--image-format", choices=["JPEG", "PNG"], default="JPEG")
    parser.add_argument("--images-per-shard", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10000)
    return parser

def validate_args(args):
    '''
    Validate preprocessing CLI arguments.
    Args:
        args (argparse.Namespace): Parsed CLI arguments.
    Returns:
        None
    '''
    if args.size <= 0:
        raise ValueError("--size must be positive")
    if args.images_per_shard <= 0:
        raise ValueError("--images-per-shard must be positive")
    if not 1 <= args.quality <= 100:
        raise ValueError("--quality must be between 1 and 100")
    if args.limit_per_split is not None and args.limit_per_split < 0:
        raise ValueError("--limit-per-split must be non-negative")

def prepare_output_dir(output_dir, overwrite):
    '''
    Validate and create the output directory.
    Args:
        output_dir (Path): Processed cache directory.
        overwrite (bool): Whether to overwrite an existing cache.
    Returns:
        None
    '''
    manifest_path = output_dir / "manifest.json"
    shards_dir = output_dir / "shards"
    existing_shards = shards_dir.exists() and any(shards_dir.glob("*.zip"))
    if not overwrite and (manifest_path.exists() or existing_shards):
        raise FileExistsError(f"Processed ImageNet cache already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

def main(argv=None):
    '''
    Preprocess ImageNet to deterministic clean RGB image shards.
    Args:
        argv (list[str] | None): Optional command-line arguments.
    Returns:
        Path: Manifest path for the processed cache.
    '''
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    data_dir = Path(args.data_dir)
    if args.output_dir is None:
        output_dir = default_output_dir(data_dir, args.size)
    else:
        output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    split_lengths = {}
    next_index = 0
    n_train_images, next_index = preprocess_split("train", data_dir, output_dir, next_index, args)
    split_lengths["train"] = n_train_images
    n_validation_images, next_index = preprocess_split("validation", data_dir, output_dir, next_index, args)
    split_lengths["validation"] = n_validation_images
    return write_manifest(output_dir, args, split_lengths)

if __name__ == "__main__":
    main()
