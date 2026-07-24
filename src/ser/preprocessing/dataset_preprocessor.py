from __future__ import annotations
from pathlib import Path
import pandas as pd
import soundfile as sf
from .audio_loader import AudioLoader
from .audio_preprocessor import AudioPreprocessor
from .common import AudioData

class DatasetPreprocessor:
    """
    Orchestrates preprocessing for an entire dataset.

    This class processes every audio file listed in the
    metadata and saves the processed audio to the output
    directory.

    Notes
    -----
    This class does not:
    - implement audio preprocessing algorithms
    - modify raw dataset
    - perform dataset splitting
    - perform data augmentation
    """

    def __init__(self, metadata: pd.DataFrame, input_root: Path, output_root: Path, metadata_output: Path):
        self.metadata = metadata.copy()
        self.input_root = input_root
        self.output_root = output_root
        self.metadata_output = metadata_output
        
        self.loader = AudioLoader()
        self.preprocessor = AudioPreprocessor()

    def process(self):
        metadata = self._clean_metadata()
        self._save_metadata(metadata)

        processed = 0
        failed = 0
        
        print(f"Total files: {len(metadata)}")
        for i, row in enumerate(metadata.itertuples(index=False), start=1):
            print(f"[{i}/{len(metadata)}] [{row.dataset}] {row.filename}")
            try:
                self._process_file(row)
                processed += 1
                print("Saved.")

            except Exception as e:
                failed += 1
                print(f"Failed: {e}")

        print(f"Processed : {processed}")
        print(f"Failed    : {failed}")

    def _clean_metadata(self) -> pd.DataFrame:
        metadata = self.metadata.copy()

        metadata.loc[
            (metadata["dataset"]=="tess")
            &(metadata["speaker"]=="OA"),
            "speaker",
        ]="OAF"
        metadata = metadata[metadata["filename"] != "mbaz_h138.wav"].copy()
        metadata = metadata[metadata["emotion"] != "Calm"].copy()

        return metadata
    
    def _build_output_path(self, filepath: Path) -> Path:
        output_path = self.output_root/filepath
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return output_path
    
    def _save_audio(self, audio_data: AudioData, output_path: Path):
        print(output_path)
        sf.write(file=output_path, data=audio_data.audio, samplerate=audio_data.sample_rate)

    def _process_file(self, row):
        input_path = self.input_root/Path(row.filepath)
        audio_data = self.loader.load(input_path)
        processed_audio = self.preprocessor.process(audio_data)
        output_path = self._build_output_path(Path(row.filepath))
        self._save_audio(processed_audio, output_path)

    def _save_metadata(self, metadata: pd.DataFrame):
        self.metadata_output.parent.mkdir(parents=True, exist_ok=True)

        metadata.to_csv(self.metadata_output, index=False)

        print(f"Metadata saved: {self.metadata_output}")