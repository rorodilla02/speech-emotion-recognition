from collections import (
    defaultdict,
    Counter
)
import statistics
from ..datasets.common import (
    ParseResult,
    DatasetStatistics,
)

class StatisticsGenerator:
    def generate(self, result: ParseResult) -> list[DatasetStatistics]:
        grouped_metadata = defaultdict(list)

        # Kelompokkan metadata berdasarkan dataset
        for metadata in result.metadata:
            grouped_metadata[metadata.dataset].append(metadata)

        statistics_list = []

        # Hitung statistik tiap dataset
        for dataset_name, metadata_list in grouped_metadata.items():
            total_files = len(metadata_list)
            speakers = sorted(
                {
                    metadata.speaker
                    for metadata in metadata_list
                }
            )
            
            total_speakers = len(speakers)
            sample_rates = sorted(
                {
                    metadata.sample_rate
                    for metadata in metadata_list
                }
            )

            durations = [
                metadata.duration
                for metadata in metadata_list
            ]
            min_duration =min(durations)
            max_duration = max(durations)
            mean_duration = statistics.mean(durations)

            emotion_distribution = dict(
                Counter(
                    metadata.emotion
                    for metadata in metadata_list
                )
            )

            dataset_statistics = DatasetStatistics(
                dataset=dataset_name,
                total_files=total_files,

                speakers=speakers,
                total_speakers=total_speakers,

                sample_rates=sample_rates,

                min_duration=min_duration,
                max_duration=max_duration,
                mean_duration=mean_duration,

                emotion_distribution=emotion_distribution,
            )

            statistics_list.append(dataset_statistics)

        return statistics_list