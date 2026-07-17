from pathlib import Path
from .common import (
    AudioMetadata,
    ParseErrorType,
    FailedFile,
    ParseResult,
    INESCO_EMOTION_MAP,
)

class INESCOParser:
    DATASET_NAME = "inesco"
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
    
    def parse(self) -> ParseResult:
        metadata_list = []
        failed_files = []
        for audio_path in self.dataset_path.rglob("*.wav"):
            try:
                filename = audio_path.stem
                parts = filename.split("_")
                
                # Validasi format nama file INESCO
                if len(parts) != 2:
                    failed = FailedFile(
                        dataset=self.DATASET_NAME,
                        filepath=audio_path,
                        filename=audio_path.name,
                        error_type=ParseErrorType.INVALID_FILENAME,
                        error_message=f"Expected 2 parts but got {len(parts)}",
                    )
                    failed_files.append(failed)
                    
                    continue

                speaker = parts[0]
                code = parts[1]

                if len(code) < 4:
                    failed_files.append(
                        FailedFile(
                            dataset=self.DATASET_NAME,
                            filepath=audio_path,
                            filename=audio_path.name,
                            error_type=ParseErrorType.INVALID_FILENAME,
                            error_message=f"Invalid emotion code format: {code}",
                        )
                    )
                    
                    continue

                emotion_code = code[0]
                sentence_number = code[1:]

                if not sentence_number.isdigit():
                    failed_files.append(
                        FailedFile(
                            dataset=self.DATASET_NAME,
                            filepath=audio_path,
                            filename=audio_path.name,
                            error_type=ParseErrorType.INVALID_FILENAME,
                            error_message=f"Recording number must be numeric, got '{sentence_number}'",
                        )
                    )
                    continue

                # Validasi Kode Emosi
                if emotion_code not in INESCO_EMOTION_MAP:
                    failed = FailedFile(
                        dataset=self.DATASET_NAME,
                        filepath=audio_path,
                        filename=audio_path.name,
                        error_type=ParseErrorType.UNKNOWN_EMOTION,
                        error_message=f"Unknown emotion code: {emotion_code}",
                    )
                    failed_files.append(failed)
                    
                    continue

                emotion = INESCO_EMOTION_MAP[emotion_code]

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