import logging
import math
from typing import List, Tuple

import PIL
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np

from fibsem.patterning import (
    FibsemMillingStage,
    FiducialPattern,
    MicroExpansionPattern,
    RectanglePattern,
    TrenchPattern,
    WaffleNotchPattern,
    BitmapPattern,
)
from fibsem.structures import (
    FibsemImage,
    FibsemImageMetadata,
    FibsemBitmapSettings,
    ImageSettings,
    Point,
    FibsemRectangleSettings,
)

COLOURS = [
    "yellow",
    "cyan",
    "magenta",
    "lime",
    "orange",
    "hotpink",
    "green",
    "blue",
    "red",
    "purple",
]


PROPERTIES = {
    "line_width": 1,
    "opacity": 0.3,
    "crosshair_size": 20,
    "crosshair_colour": "yellow",
    "rotation_point": "center",
}


def generate_blank_image(
    resolution: List[int] = [1536, 1024],
    hfw: float = 100e-6,
    pixel_size: Point = None,
) -> FibsemImage:
    """Generate a blank image with a given resolution and field of view.
    Args:
        resolution: List[int]: Resolution of the image.
        hfw: float: Horizontal field width of the image.
        pixel_size: Point: Pixel size of the image.
    Returns:
        FibsemImage: Blank image with valid metadata from display.
    """
    # need at least one of hfw, pixelsize
    if pixel_size is None and hfw is None:
        raise ValueError("Need to specify either hfw or pixelsize")

    if pixel_size is None:
        vfw = hfw * resolution[1] / resolution[0]
        pixel_size = Point(hfw / resolution[0], vfw / resolution[1])

    image = FibsemImage(
        data=np.zeros((resolution[1], resolution[0]), dtype=np.uint8),
        metadata=FibsemImageMetadata(
            image_settings=ImageSettings(hfw=hfw, resolution=resolution),
            microscope_state=None,
            pixel_size=pixel_size,
        ),
    )
    return image


def _rect_pattern_to_image_pixels(
    pattern: FibsemRectangleSettings, pixel_size: float, image_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """Convert rectangle pattern to image pixel coordinates.
    Args:
        pattern: FibsemRectangleSettings: Rectangle pattern to convert.
        pixel_size: float: Pixel size of the image.
        image_shape: Tuple[int, int]: Shape of the image.
    Returns:
        Tuple[int, int, int, int]: Parameters (center_x, center_y, width, height) in image pixel coordinates.
    """
    # get pattern parameters
    width = pattern.width
    height = pattern.height
    mx, my = pattern.centre_x, pattern.centre_y

    # position in metres from image centre
    pmx, pmy = mx / pixel_size, my / pixel_size

    # convert to image coordinates
    cy, cx = image_shape[0] // 2, image_shape[1] // 2
    px = cx + pmx
    py = cy - pmy

    # convert parameters to pixels
    width = width / pixel_size
    height = height / pixel_size

    return px, py, width, height


# TODO: circle patches, line patches
def _draw_rectangle_pattern(
    image: FibsemImage,
    pattern: RectanglePattern,
    colour: str = "yellow",
    name: str = "Rectangle",
) -> List[PatchCollection]:
    """Draw a rectangle pattern on an image.
    Args:
        image: FibsemImage: Image to draw pattern on.
        pattern: RectanglePattern: Rectangle pattern to draw.
        colour: str: Colour of rectangle patches.
        name: str: Name of the rectangle patches.
    Returns:
        List[mpatches.Rectangle]: List of patches to draw.
    """
    # common image properties
    pixel_size = image.metadata.pixel_size.x  # assume isotropic
    image_shape = image.data.shape

    patches = []
    for i, p in enumerate(pattern.patterns, 1):
        # convert from microscope image (real-space) to image pixel-space
        px, py, width, height = _rect_pattern_to_image_pixels(
            p, pixel_size, image_shape
        )

        patch_collection = PatchCollection(
            [
                mpatches.Rectangle(
                    (px - width / 2, py - height / 2),  # bottom left corner
                    width=width,
                    height=height,
                    angle=math.degrees(p.rotation),
                    rotation_point=PROPERTIES["rotation_point"],
                    linewidth=PROPERTIES["line_width"],
                    edgecolor=colour,
                    facecolor=colour,
                    alpha=PROPERTIES["opacity"],
                )
            ],
            match_original=True,
        )
        if i == 1:
            patch_collection.set_label(f"{name}")
        patches.append(patch_collection)

    return patches


def _draw_bitmap_pattern(
    image: FibsemImage,
    pattern: FibsemBitmapSettings,
    colour: str = "yellow",
    name: str = "Bitmap",
) -> List[PatchCollection]:
    """Draw a rectangle pattern on an image.
    Args:
        image: FibsemImage: Image to draw pattern on.
        pattern: RectanglePattern: Rectangle pattern to draw.
        colour: str: Colour of rectangle patches.
        name: str: Name of the rectangle patches.
    Returns:
        List[mpatches.Rectangle]: List of patches to draw.
    """
    # common image properties
    pixel_size = image.metadata.pixel_size.x  # assume isotropic
    image_shape = image.data.shape

    patches = []
    for i, p in enumerate(pattern.patterns, 1):
        # convert from microscope image (real-space) to image pixel-space
        px, py, width, height = _rect_pattern_to_image_pixels(
            p, pixel_size, image_shape
        )
        pixel_positions = tuple(range(int(round(width))))

        array = np.asarray(PIL.Image.open(p.path, formats=("BMP",))).flatten()
        interp_array = np.interp(pixel_positions, range(array.size), array)

        bitmap_rects = []
        for j in pixel_positions:
            # Draw a thin rectangle for each bitmap pixel (assumed to be 1D)
            bitmap_rects.append(
                mpatches.Rectangle(
                    (
                        px - (width / 2) + j,
                        py - height / 2,
                    ),  # bottom left corner
                    width=1,
                    height=height,
                    angle=math.degrees(p.rotation),
                    rotation_point=PROPERTIES["rotation_point"],
                    linewidth=0,
                    edgecolor="none",
                    facecolor=colour,
                    alpha=interp_array[j] / 255,
                )
            )
        # Draw the edges
        bitmap_rects.append(
            mpatches.Rectangle(
                (
                    px - width / 2,
                    py - height / 2,
                ),  # bottom left corner
                width=width,
                height=height,
                angle=math.degrees(p.rotation),
                rotation_point=PROPERTIES["rotation_point"],
                linewidth=PROPERTIES["line_width"],
                edgecolor=colour,
                facecolor="none",
                alpha=PROPERTIES["opacity"],
            )
        )

        # Store all the rectangles as a patch collection
        patch_collection = PatchCollection(bitmap_rects, match_original=True)

        if i == 1:
            patch_collection.set_label(f"{name}")
        patches.append(patch_collection)

    return patches


drawing_functions = {
    RectanglePattern: _draw_rectangle_pattern,
    TrenchPattern: _draw_rectangle_pattern,
    MicroExpansionPattern: _draw_rectangle_pattern,
    FiducialPattern: _draw_rectangle_pattern,
    WaffleNotchPattern: _draw_rectangle_pattern,
    BitmapPattern: _draw_bitmap_pattern,
}


def draw_milling_patterns(
    image: FibsemImage,
    milling_stages: List[FibsemMillingStage],
    crosshair: bool = True,
    scalebar: bool = True,
) -> plt.Figure:
    """
    Draw milling patterns on an image.
    Args:
        image: FibsemImage: Image to draw patterns on.
        milling_stages: List[FibsemMillingStage]: Milling stages to draw.
        crosshair: bool: Draw crosshair at centre of image.
        scalebar: bool: Draw scalebar on image.
    Returns:
        plt.Figure: Figure with patterns drawn.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image.data, cmap="gray")

    patch_collections = []
    for i, stage in enumerate(milling_stages):
        colour = COLOURS[i % len(COLOURS)]
        p = stage.pattern

        patch_collections.extend(
            drawing_functions[type(p)](image, p, colour=colour, name=stage.name)
        )

    for pc in patch_collections:
        ax.add_collection(pc)
    ax.legend()

    # draw crosshair at centre of image
    if crosshair:
        cy, cx = image.data.shape[0] // 2, image.data.shape[1] // 2
        ax.plot(cx, cy, "y+", markersize=PROPERTIES["crosshair_size"])

    # draw scalebar
    if scalebar:
        try:
            # optional dependency, best effort
            from matplotlib_scalebar.scalebar import ScaleBar

            scalebar = ScaleBar(
                dx=image.metadata.pixel_size.x,
                color="black",
                box_color="white",
                box_alpha=0.5,
                location="lower right",
            )

            plt.gca().add_artist(scalebar)
        except ImportError:
            logging.debug("Scalebar not available, skipping")

    return fig, ax
