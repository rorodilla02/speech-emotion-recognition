import soundfile as sf
from ..datasets.common import AudioMetadata

class AudioReader:
    def __init__(self):
        pass

    def read(self, metadata: AudioMetadata) -> AudioMetadata:
        try:
            info = sf.info(metadata.filepath)

            metadata.sample_rate = info.samplerate
            metadata.duration = info.duration

            return metadata
        
        except Exception as e:
            raise RuntimeError(
                f"Failed to read audio metadata: {metadata.filepath} ({e})"
            ) from e