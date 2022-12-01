from abc import ABC, abstractmethod
import fibsem.utils as utils
from pathlib import Path
import fibsem.calibration as calibration
import os
import logging
from autoscript_sdb_microscope_client.structures import GrabFrameSettings
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem.structures import BeamType, GammaSettings, ImageSettings, ReferenceImages, FibsemImage


class FibsemMicroscope(ABC):
    """Abstract class containing all the core microscope functionalities"""    
    @abstractmethod
    def connect(self, host: str, port: int):
        pass

    @abstractmethod
    def disconnect(self):
        pass


class ThermoMicroscope(FibsemMicroscope):
    """ThermoFisher Microscope class, uses FibsemMicroscope as blueprint 

    Args:
        FibsemMicroscope (ABC): abstract implementation
    """
    def __init__(self):
        self.connection = SdbMicroscopeClient()

    def disconnect(self):
        self.connection.disconnect()
        pass
    
    # @classmethod
    def connect_to_microscope(self, ip_address: str, port: int = 7520) -> None:
        """Connect to a Thermo Fisher microscope at a specified I.P. Address and Port
        
        Args:
            ip_address (str): I.P. Address of microscope 
            port (int): port of microscope (default: 7520)
            """
        try:
            # TODO: get the port
            logging.info(f"Microscope client connecting to [{ip_address}:{port}]")
            self.connection.connect(host=ip_address, port=port)
            logging.info(f"Microscope client connected to [{ip_address}:{port}]")
        except Exception as e:
            logging.error(f"Unable to connect to the microscope: {e}")
            
    def acquire_image(self, frame_settings: GrabFrameSettings = None, image_settings = ImageSettings) -> FibsemImage:
        """Acquire a new image.
        
        Args:
            settings (GrabFrameSettings, optional): frame grab settings. Defaults to None.
            beam_type (BeamType, optional): imaging beam type. Defaults to BeamType.ELECTRON.
        
        Returns:
            AdornedImage: new image
        """
        logging.info(f"acquiring new {image_settings.beam_type.name} image.")
        self.connection.imaging.set_active_view(image_settings.beam_type.value)
        self.connection.imaging.set_active_device(image_settings.beam_type.value)
        image = self.connection.imaging.grab_frame(frame_settings)
        
        state = calibration.get_current_microscope_state(self.connection)
        image = FibsemImage.fromAdornedImage(image, image_settings, state)        
        return image

    def last_image(self, beam_type: BeamType =BeamType.ELECTRON) -> FibsemImage:
        """Get the last previously acquired image.

        Args:
            microscope (SdbMicroscopeClient):  autoscript microscope instance
            beam_type (BeamType, optional): imaging beam type. Defaults to BeamType.ELECTRON.

        Returns:
            AdornedImage: last image
        """

        self.connection.imaging.set_active_view(beam_type.value)
        self.connection.imaging.set_active_device(beam_type.value)
        image = self.connection.imaging.get_image()
        
        state = calibration.get_current_microscope_state(self.connection)
        
        image_settings = ImageSettings(
            resolution=f"{image.width}x{image.height}",
            dwell_time=image.metadata.scan_settings.dwell_time,
            hfw=image.width * image.metadata.binary_result.pixel_size.x,
            autocontrast=True,
            beam_type=BeamType.ELECTRON,
            gamma=GammaSettings(),
            save=False,
            save_path="path",
            label=utils.current_timestamp(),
            reduced_area=None,
            )
        
        fibsem_img = FibsemImage.fromAdornedImage(image, image_settings, state)
        return fibsem_img     
        
        
        