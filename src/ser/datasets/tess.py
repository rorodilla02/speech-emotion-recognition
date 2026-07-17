from pathlib import Path
from .common import (
    AudioMetadata,
    ParseErrorType,
    FailedFile,
    ParseResult,
    TESS_EMOTION_MAP,
)

class TESSParser:
    DATASET_NAME = "tess"
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
    
    def parse(self) -> ParseResult:
        metadata_list = []
        failed_files = []
        for audio_path in self.dataset_path.rglob("*.wav"):
            try:
                filename = audio_path.stem
                parts = filename.split("_")
                
                # Validasi format nama file TESS
                if len(parts) != 3:
                    failed = FailedFile(
                        dataset=self.DATASET_NAME,
                        filepath=audio_path,
                        filename=audio_path.name,
                        error_type=ParseErrorType.INVALID_FILENAME,
                        error_message=f"Expected 3 parts but got {len(parts)}",
                    )
                    failed_files.append(failed)
                    
                    continue

                speaker = parts[0]
                emotion_code = parts[2]

                # Validasi Kode Emosi
                if emotion_code not in TESS_EMOTION_MAP:
                    failed = FailedFile(
                        dataset=self.DATASET_NAME,
                        filepath=audio_path,
                        filename=audio_path.name,
                        error_type=ParseErrorType.UNKNOWN_EMOTION,
                        error_message=f"Unknown emotion code: {emotion_code}",
                    )
                    failed_files.append(failed)
                    
                    continue

                emotion = TESS_EMOTION_MAP[emotion_code]

                metadata = AudioMetadata(
                    dataset=self.DATASET_NAME,
                    filepath=audio_path,
                    filename=audio_path.name,
                    speaker=speaker,
                    raw_label=emotion_code,
                    emotion=emotion,
                )
                metadata_list.append(metadata)
            
            except Exception as e:
                failed = FailedFile(
                    dataset=self.DATASET_NAME,
                    filepath=audio_path,
                    filename=audio_path.name,
                    error_type=ParseErrorType.UNEXPECTED_ERROR,
                    error_message=str(e),
                )
                failed_files.append(failed)

        return ParseResult(
            metadata=metadata_list,
            failed_files=failed_files,
        )