import json
import logging
import multiprocessing as mp
from pathlib import Path

import click
import cv2
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from utils import Compression
from utils import DataType
from utils import FaceForensicsDataStructure

logger = logging.getLogger(__file__)


def _extract_face(img_path, face, face_images_dir):
    if not face:
        return False
    img = cv2.imread(str(img_path))
    y1, x1, y2, x2 = face[0][0],face[0][1],face[0][2],face[0][3]
    #print("YES")
    cropped_face = img[y1 : y2, x2 : x1]  # noqa E203
    cv2.imwrite(str(face_images_dir / img_path.name), cropped_face)
    return True


def _extract_faces_from_video(
    video_folder: Path, bounding_boxes: Path, face_images: Path
) -> bool:
    with open(str((bounding_boxes / video_folder.name).with_suffix(".json")), "r") as f:
        faces = json.load(f)

    face_images = face_images / video_folder.name
    face_images.mkdir(exist_ok=True)

    # extract all faces and save it
    for img in sorted(video_folder.iterdir()):
        face = faces[img.with_suffix("").name]
        _extract_face(img, face, face_images)

    return True


@click.command()
@click.option("--source_dir_root", required=False,default='/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset', type=click.Path(exists=True))
@click.option("--compressions", "-c", default=[Compression.c40])
@click.option(
    "--methods", "-m", multiple=True, default=FaceForensicsDataStructure.ALL_METHODS
)
@click.option("--cpu_count", required=False, type=click.INT, default=mp.cpu_count())
def extract_faces(source_dir_root, compressions, methods, cpu_count):
    full_images_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        compressions=compressions,
        data_types=(DataType.full_images,),
        methods=methods,
    )

    bounding_boxes_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        compressions=compressions,
        data_types=(DataType.bounding_boxes,),
        methods=methods,
    )

    face_images_dir_data_structure = FaceForensicsDataStructure(
        source_dir_root,
        compressions=compressions,
        data_types=(DataType.face_images,),
        methods=methods,
    )

    for full_images, bounding_boxes, face_images in zip(
        full_images_data_structure.get_subdirs(),
        bounding_boxes_dir_data_structure.get_subdirs(),
        face_images_dir_data_structure.get_subdirs(),
    ):
        logger.info(f"Current method: {full_images.parents[1].name}")

        face_images.mkdir(exist_ok=True)
        tey=sorted(full_images.iterdir())
        # extract faces from videos in parallel
        Parallel(n_jobs=cpu_count)(
            delayed(
                lambda _video_folder: _extract_faces_from_video(
                    _video_folder, bounding_boxes, face_images
                )
            )(video_folder)
            for video_folder in tqdm(sorted(full_images.iterdir()))
        )


if __name__ == "__main__":
    extract_faces()
