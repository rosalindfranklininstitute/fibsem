import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# fibsem
from fibsem.milling.base import (
    MillingStrategy,
    MillingStrategyConfig,
    FibsemMillingStage
)
from fibsem.milling import setup_milling, finish_milling
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.patterning.patterns2 import (
    TrenchPattern,
    TrenchBitmapPattern,
)

# adaptive polish
from adaptive_polish.main import AdaptivePolish

@dataclass
class AdaptivePolishMillingConfig(MillingStrategyConfig):
    align_sem: bool = True
    milling_interval_s: int = 10
    gis_stop_um: float = 0.2
    max_crack_area_um2: float = 2
    max_milling_cycles: int = 30
    window_size_px: int = 10
    model_path: str = "abc"  #TODO add support in FibsemMillingWidget.set_milling_strategy_ui for path inputs

    _advanced_attributes = []

    @staticmethod
    def from_dict(d: dict) -> "AdaptivePolishMillingConfig":
        return AdaptivePolishMillingConfig(**d)

    def to_dict(self):
        return {
            "milling_interval_s": self.milling_interval_s,
            "gis_stop_m": self.gis_stop_um,
            "max_crack_area_m2": self.max_crack_area_um2,
            "max_milling_cycles": self.max_milling_cycles,
            "use_sem_beam_shift_alignment": self.align_sem,
            "window_size_px": self.window_size_px,
            "model_path": self.model_path,
        }


@dataclass
class AdaptivePolishMillingStrategy(MillingStrategy):
    name: str = "Adaptive Polish"
    fullname: str = "Adaptive polishing according to GIS thickness"

    def __init__(self, config: AdaptivePolishMillingConfig = None):
        self.config = config or AdaptivePolishMillingConfig()

    def to_dict(self):
        return {"name": self.name, "config": self.config.to_dict()}

    @staticmethod
    def from_dict(d: dict) -> "AdaptivePolishMillingStrategy":
        config = AdaptivePolishMillingConfig.from_dict(d["config"])
        return AdaptivePolishMillingStrategy(config=config)

    def run(
        self,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        asynch: bool = False,  # what does this do
        parent_ui = None,  # what does this do
    ) -> None:
        """Run adaptive polishing
        #TODO: improve docs

        Args:
            microscope (FibsemMicroscope): See `fibsem.microscope.FibsemMicroscope`
            stage (FibsemMillingStage): See `fibsem.milling.base.FibsemMillingStage`
            asynch (bool, optional): Run asynchronously? Defaults to False.
            parent_ui (_type_, optional): Napari UI. Defaults to None.
        """
        logging.info(f"Running {self.fullname} for {stage.name}")

        # setup milling
        ap = AdaptivePolish(self.config.to_dict(), microscope)
        setup_milling()

        # run adaptive polishing
        ap.adaptive_polish_run(
            microscope=microscope,
            stage=stage,
        )

        # finish milling (clear patterns, restore imaging current)
        finish_milling(
            microscope=microscope,
            imaging_current=microscope.system.ion.beam.beam_current,
            imaging_voltage=microscope.system.ion.beam.voltage,
        )
