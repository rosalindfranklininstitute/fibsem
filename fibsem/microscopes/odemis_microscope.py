
import sys
def add_odemis_path():
    """Add the odemis path to the python path"""
    def parse_config(path) -> dict:
        """Parse the odemis config file and return a dict with the config values"""
        
        with open(path) as f:
            config = f.read()

        config = config.split("\n")
        config = [line.split("=") for line in config]
        config = {line[0]: line[1].replace('"', "") for line in config if len(line) == 2}
        return config

    odemis_path = "/etc/odemis.conf"
    config = parse_config(odemis_path)
    sys.path.append(f"{config['DEVPATH']}/odemis/src")  # dev version
    sys.path.append("/usr/lib/python3/dist-packages")   # release version + pyro4

add_odemis_path()

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, FibsemBitmapSettings, FibsemCircleSettings, FibsemImage, FibsemLineSettings, FibsemManipulatorPosition, FibsemRectangleSettings, ImageSettings, FibsemStagePosition
from odemis import model
from odemis.acq.stream import SEMStream, FIBStream
from odemis.acq.acqmng import acquire

def stage_position_to_odemis_dict(position: FibsemStagePosition) -> dict:
    """Convert a FibsemStagePosition to a dict with the odemis keys"""
    pdict = position.to_dict()
    pdict.pop("name")
    pdict.pop("coordinate_system")
    pdict["rz"] = pdict.pop("r")
    pdict["rx"] = pdict.pop("t")

    # if any values are None, remove them
    pdict = {k: v for k, v in pdict.items() if v is not None}

    return pdict

def odemis_dict_to_stage_position(pdict: dict) -> FibsemStagePosition:
    """Convert a dict with the odemis keys to a FibsemStagePosition"""
    pdict["r"] = pdict.pop("rz")
    pdict["t"] = pdict.pop("rx")
    pdict["coordinate_system"] = "RAW"
    return FibsemStagePosition.from_dict(pdict)




class OdemisMicroscope(FibsemMicroscope):

    def __init__(self):

        self.connection = model.getComponent(role="fibsem")

        # setup electron beam, det
        electron_beam = model.getComponent(role="e-beam")
        electron_det = model.getComponent(role="electron-detector")

        # setup ion beam, det
        ion_beam = model.getComponent(role="ion-beam")
        ion_det = model.getComponent(role="ion-detector")

        # create streams
        self.sem_stream = SEMStream("sem-stream", electron_det, electron_det.data, electron_beam)
        self.fib_stream = FIBStream("fib-stream", ion_det, ion_det.data, ion_beam)

    def connect_to_microscope(self, ip_address: str, port: int) -> None:
        pass

    def disconnect(self):
        pass

    def acquire_chamber_image(self) -> FibsemImage:
        pass

    def acquire_image(self, image_settings: ImageSettings) -> FibsemImage:
        
        beam_type = image_settings.beam_type

        if beam_type is BeamType.ELECTRON:
            stream = self.sem_stream
        if beam_type is BeamType.ION:
            stream = self.fib_stream

        f = acquire([stream])
        data, ex = f.result(timeout=600)

        return FibsemImage(data[0], None) # TODO: metadata


    def last_image(self, beam_type: BeamType) -> FibsemImage:
        pass

    def autocontrast(self, beam_type: BeamType) -> None:
        pass
    
    def auto_focus(self, beam_type: BeamType) -> None:
        pass

    def beam_shift(self, dx: float, dy: float, beam_type: BeamType) -> None:
        pass

    def _get(self, key: str, beam_type: BeamType = None) -> str:
        
        if key == "stage_position":
            stage = model.getComponent(role="stage-bare")
            pdict = stage.position.value
            value = odemis_dict_to_stage_position(pdict)
        return value

    def _set(self, key: str, value: str, beam_type: BeamType = None) -> None:
        pass

    def _get_saved_manipulator_position(self, name: str) -> FibsemManipulatorPosition:
        pass

    def get_available_values(self, key: str) -> list:
        pass

    def check_available_values(self, key: str) -> list:
        pass

    def insert_manipulator(self) -> None:
        pass

    def move_manipulator_absolute(self, position: FibsemManipulatorPosition) -> None:
        pass

    def move_manipulator_relative(self, position: FibsemManipulatorPosition) -> None:
        pass

    def move_manipulator_corrected(self, position: FibsemManipulatorPosition) -> None:
        pass

    def move_manipulator_to_position_offset(self, offset: FibsemManipulatorPosition, name: str) -> None:
        pass

    def retract_manipulator(self) -> None:
        pass

    def move_stage_absolute(self, position: FibsemStagePosition) -> None:
        stage = model.getComponent(role="stage-bare")
        pdict = stage_position_to_odemis_dict(position)
        stage.moveAbsSync(pdict)

    def move_stage_relative(self, position: FibsemStagePosition) -> None:
        stage = model.getComponent(role="stage-bare")
        pdict = stage_position_to_odemis_dict(position)
        stage.moveRelSync(pdict)

    def stable_move(self, dx: float, dy: float, beam_type: BeamType) -> None:
        pass

    def vertical_move(self, dx: float, dy: float) -> None:
        pass

    def safe_absolute_stage_movement(self, position: FibsemStagePosition) -> None:
        pass

    def live_imaging(self, beam_type: BeamType) -> None:
        pass

    def consume_image_queue(self):
        pass

    def draw_bitmap_pattern(self, pattern_settings: FibsemBitmapSettings, path: str):
        pass

    def draw_rectangle(self, pattern_settings: FibsemRectangleSettings):
        pdict = pattern_settings.to_dict()
        print(pdict)

        pdict["center_x"] = pdict.pop("centre_x")
        pdict["center_y"] = pdict.pop("centre_y")

        self.connection.create_rectangle(pdict)

    def draw_circle(self, pattern_settings: FibsemCircleSettings):
        
        pass

    def draw_line(self, pattern_settings: FibsemLineSettings):
        pass

    def setup_milling(self, milling_current: float, milling_voltage: float):
        pass       
        
    def setup_sputter(self):
        pass

    def draw_sputter_pattern(self):
        pass

    def run_sputter(self, *args, **kwargs):
        pass

    def finish_sputter(self):
        pass

    def run_milling(self, milling_current: float, milling_voltage: float, asynch: bool = False):
        
        if asynch:
            self.connection.start_milling()
        else:
            self.connection.run_milling()

    def get_milling_state(self):
        from fibsem.structures import PatterningState
        return PatterningState[self.connection.get_patterning_state().upper()]

    def run_milling_drift_corrected(self):
        pass

    def finish_milling(self, imaging_current: float, imaging_voltage: float) -> None:
        
        self.connection.clear_patterns()

    def estimate_milling_time(self) -> float:
        
        return self.connection.estimate_milling_time()