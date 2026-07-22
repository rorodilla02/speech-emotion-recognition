from pathlib import Path
from ser.datasets.ravdess import RAVDESSParser
from ser.datasets.tess import TESSParser
from ser.datasets.savee import SAVEEParser
from ser.datasets.inesco import INESCOParser
from ser.utils.audio import AudioReader
from ser.audit.auditor import DatasetAuditor
from ser.statistics.generator import StatisticsGenerator
from ser.output.writer import OutputWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

def main():
    parsers = [
        RAVDESSParser(RAW_DIR / "ravdess"),
        TESSParser(RAW_DIR / "tess"),
        SAVEEParser(RAW_DIR / "savee"),
        INESCOParser(RAW_DIR / "inesco"),
    ]

    audio_reader = AudioReader()
    auditor = DatasetAuditor(audio_reader=audio_reader, raw_root=RAW_DIR)
    statistics_generator = StatisticsGenerator()
    writer = OutputWriter(METADATA_DIR)
    
    print("Starting dataset audit...")
    result = auditor.audit(parsers)
    print(f"Parsed {len(result.metadata)} files {len(result.failed_files)} failed")
    print("Generating statistics...")
    statistics = statistics_generator.generate(result)
    print(f"Generated statistics for {len(statistics)} datasets")
    print("writing reports...")
    writer.write_all(result, statistics)
    print("Dataset audit completed successfully!")

    for stats in statistics:
        print(stats.dataset)
        print(stats.speakers)
    
if __name__ == "__main__":
    main()