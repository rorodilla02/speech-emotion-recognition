from collections import (
    defaultdict,
    Counter,
)
import numpy as np
from ..datasets.common import (
    ParseResult,
    DatasetStatistics,
)


class StatisticsGenerator:
    """
    Menghitung statistik deskriptif per dataset dari hasil audit.

    Catatan
    -------
    Kelas ini tidak:
    - membaca file audio
    - menulis file output
    - memodifikasi metadata
    """

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
                    if metadata.sample_rate is not None
                }
            )

            durations = np.asarray(
                [
                    metadata.duration
                    for metadata in metadata_list
                    if metadata.duration is not None
                ],
                dtype=float,
            )

            if durations.size == 0:
                raise ValueError(
                    f"Dataset '{dataset_name}' tidak memiliki durasi yang valid."
                )

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

                min_duration=self._round(durations.min()),
                max_duration=self._round(durations.max()),
                mean_duration=self._round(durations.mean()),
                # ddof=1 agar konsisten dengan pandas.Series.std() di DurationAnalyzer
                std_duration=self._round(
                    durations.std(ddof=1) if durations.size > 1 else 0.0
                ),
                p5_duration=self._round(np.percentile(durations, 5)),
                p95_duration=self._round(np.percentile(durations, 95)),

                emotion_distribution=emotion_distribution,
            )

            statistics_list.append(dataset_statistics)

        return statistics_list

    @staticmethod
    def _round(value: float) -> float:
        return round(float(value), 3)