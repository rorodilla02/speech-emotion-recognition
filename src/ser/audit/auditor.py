from ..utils.audio import AudioReader
from ..datasets.common import (
    ParseResult,
    BaseParser,
    FailedFile,
    ParseErrorType,
)

class DatasetAuditor:
    def __init__(self, audio_reader: AudioReader):
        self.audio_reader = audio_reader

    def audit(self, parsers: list[BaseParser]) -> ParseResult:
        metadata_list = []
        failed_files = []

        # Parse metadata dari seluruh dataset
        for parser in parsers:
            result = parser.parse()

            metadata_list.extend(result.metadata)
            failed_files.extend(result.failed_files)
        
        valid_metadata = []

        # Baca informasi audio
        for metadata in metadata_list:
            try:
                metadata = self.audio_reader.read(metadata)
                valid_metadata.append(metadata)

            except RuntimeError as e:
                failed = FailedFile(
                    dataset=metadata.dataset,
                    filepath=metadata.filepath,
                    filename=metadata.filename,
                    error_type=ParseErrorType.AUDIO_READ_ERROR,
                    error_message=str(e),
                )
                failed_files.append(failed)

        return ParseResult(
            metadata=valid_metadata,
            failed_files=failed_files
        )