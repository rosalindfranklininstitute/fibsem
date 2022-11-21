import numpy as np
import tifffile as tff
from dataclasses import dataclass
import json
from fibsem.structures import ImageSettings

THERMO_ENABLED = True
if THERMO_ENABLED:
    from autoscript_sdb_microscope_client.structures import AdornedImage


@dataclass
class FibsemImageMetadata:
    """Metadata for a FibsemImage."""

    image_settings: ImageSettings

    def __to_dict__(self)  -> dict:
        """Converts metadata to a dictionary.

        Returns:
            dictionary: self as a dictionary
        """
        settings_dict = self.image_settings.__to_dict__()
        return settings_dict


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
                metadata = FibsemImageMetadata(
                    image_settings=ImageSettings.__from_dict__(metadata)
                )
            except:
                metadata = None
        return cls(data=data, metadata=metadata)

    def save(self, save_path: str) -> None:
        """Saves a FibsemImage to a tiff file.

        Inputs:
            save_path (path): path to save directory and filename
        """
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
        cls, adorned: AdornedImage, metadata: FibsemImageMetadata = None
    ) -> "FibsemImage":
        """Creates FibsemImage from an AdornedImage (microscope output format).

        Args:
            adorned (AdornedImage): Adorned Image from microscope
            metadata (FibsemImageMetadata, optional): metadata extracted from microscope output. Defaults to None.

        Returns:
            FibsemImage: instance of FibsemImage from AdornedImage
        """
        return cls(data=adorned.data, metadata=metadata)

def check_data_format(data: np.ndarray) -> np.ndarray:
    """Checks that data is in the correct format."""
    assert data.ndim == 2  # or data.ndim == 3
    assert data.dtype == np.uint8
    # if data.ndim == 3 and data.shape[2] == 1:
    #     data = data[:, :, 0]
    return data