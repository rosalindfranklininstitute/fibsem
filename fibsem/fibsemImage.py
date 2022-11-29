import numpy as np
import tifffile as tff
from dataclasses import dataclass
import json
import os
from fibsem.structures import ImageSettings, BeamType, GammaSettings, Point, MicroscopeState, BeamSettings, FibsemRectangle
from fibsem.config import METADATA_VERSION

THERMO_ENABLED = True
if THERMO_ENABLED:
    from autoscript_sdb_microscope_client.structures import (AdornedImage, StagePosition)

from pathlib import Path

@dataclass
class FibsemImageMetadata:
    """Metadata for a FibsemImage."""

    image_settings: ImageSettings
    pixel_size: Point
    microscope_state: MicroscopeState
    version: str = METADATA_VERSION

    def __to_dict__(self) -> dict:
        """Converts metadata to a dictionary.

        Returns:
            dictionary: self as a dictionary
        """
        if self.image_settings is not None:
            settings_dict = self.image_settings.__to_dict__()
        if self.version is not None:
            settings_dict["version"] = self.version
        if self.pixel_size is not None:
            settings_dict["pixel_size"] = self.pixel_size.__to_dict__()
        if self.microscope_state is not None:
            settings_dict["microscope_state"] = self.microscope_state.__to_dict__()
        return settings_dict

    @staticmethod
    def __from_dict__(settings: dict) -> "ImageSettings":
        """Converts a dictionary to metadata."""

        image_settings = ImageSettings.__from_dict__(settings)
        if settings["version"] is not None:
            version = settings["version"]
        if settings["pixel_size"] is not None:
            pixel_size = Point.__from_dict__(settings["pixel_size"])
        if settings["microscope_state"] is not None:
            microscope_state = MicroscopeState(
                timestamp=settings["microscope_state"]["timestamp"],
                absolute_position=StagePosition(),
                eb_settings=BeamSettings.__from_dict__(settings["microscope_state"]["eb_settings"]),
                ib_settings=BeamSettings.__from_dict__(settings["microscope_state"]["ib_settings"]),
            )

        metadata = FibsemImageMetadata(
            image_settings=image_settings,
            version=version,
            pixel_size=pixel_size,
            microscope_state=microscope_state,
        )
        return metadata


class FibsemImage:
    def __init__(self, data: np.ndarray, metadata: FibsemImageMetadata = None):
        self.data = check_data_format(data)
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = None

    @classmethod
    def load(cls, tiff_path: str) -> "FibsemImage":
        """Loads a FibsemImage from a tiff file.

        Args:
            tiff_path (path): path to the tif* file

        Returns:
            FibsemImage: instance of FibsemImage
        """
        with tff.TiffFile(tiff_path) as tiff_image:
            data = tiff_image.asarray()
            try:
                metadata = json.loads(
                    tiff_image.pages[0].tags["ImageDescription"].value
                )
                metadata = FibsemImageMetadata.__from_dict__(metadata)
            except Exception as e:
                metadata = None
                print(f"Error: {e}")
        return cls(data=data, metadata=metadata)

    def save(self, save_path: Path) -> None:
        """Saves a FibsemImage to a tiff file.

        Inputs:
            save_path (path): path to save directory and filename
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path = Path(save_path).with_suffix(".tif")
        
        if self.metadata is not None:
            metadata_dict = self.metadata.__to_dict__()
        else:
            metadata_dict = None
        tff.imwrite(
            save_path,
            self.data,
            metadata=metadata_dict,
        )

    @classmethod
    def fromAdornedImage(
        cls, adorned: AdornedImage, image_settings: ImageSettings, state: MicroscopeState = None
    ) -> "FibsemImage":
        """Creates FibsemImage from an AdornedImage (microscope output format).

        Args:
            adorned (AdornedImage): Adorned Image from microscope
            metadata (FibsemImageMetadata, optional): metadata extracted from microscope output. Defaults to None.

        Returns:
            FibsemImage: instance of FibsemImage from AdornedImage
        """

        if state is None:
            state = MicroscopeState(
                timestamp=adorned.metadata.acquisition.acquisition_datetime,
                absolute_position=StagePosition(),
                eb_settings=BeamSettings(beam_type=BeamType.ELECTRON),
                ib_settings=BeamSettings(beam_type=BeamType.ION),
            )
        else:
            state.timestamp = adorned.metadata.acquisition.acquisition_datetime

        pixel_size = Point(adorned.metadata.binary_result.pixel_size.x, adorned.metadata.binary_result.pixel_size.y)
        metadata=FibsemImageMetadata(image_settings=image_settings, pixel_size=pixel_size, microscope_state=state)
        return cls(data=adorned.data, metadata=metadata)


def check_data_format(data: np.ndarray) -> np.ndarray:
    """Checks that data is in the correct format."""
    assert data.ndim == 2  # or data.ndim == 3
    assert data.dtype == np.uint8
    # if data.ndim == 3 and data.shape[2] == 1:
    #     data = data[:, :, 0]
    return data