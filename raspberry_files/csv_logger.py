import csv
import math
import time
from pathlib import Path


EARTH_RADIUS_M = 6_371_000.0


def distance_m(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


class LetterCsvLogger:
    def __init__(
        self,
        csv_path,
        different_letter_distance_m=25.0,
        min_confidence=0.20,
        track_timeout_s=3.0,
        min_frames=3,
    ):
        """
        csv_path:
            Путь к CSV.

        different_letter_distance_m:
            Если новая пространственная область дальше 25 м,
            это другая буква.

        min_confidence:
            Не учитывать слабые детекции.

        track_timeout_s:
            Если детекций в области не было столько секунд,
            группа закрывается, выбирается победитель и
            записывается одна строка в CSV.

        min_frames:
            Минимальное число подходящих детекций области,
            чтобы она могла быть сохранена.
        """

        self.csv_path = Path(csv_path)
        self.different_letter_distance_m = different_letter_distance_m
        self.min_confidence = min_confidence
        self.track_timeout_s = track_timeout_s
        self.min_frames = min_frames

        self.current_track = None
        self.saved_letters = []

        self._create_csv_if_needed()
        self._load_saved_letters()

    def _create_csv_if_needed(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file, delimiter=";")
                writer.writerow([
                    "letter",
                    "latitude",
                    "longitude",
                ])

    def _load_saved_letters(self):
        """
        Загружает ранее записанные буквы.
        При новом запуске скрипта та же буква ближе 25 м
        не будет добавлена повторно.
        """

        with self.csv_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file, delimiter=";")

            for row in reader:
                try:
                    self.saved_letters.append(
                        {
                            "letter": row["letter"],
                            "latitude": float(row["latitude"]),
                            "longitude": float(row["longitude"]),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue

    def add_detection(self, letter, confidence, latitude, longitude):
        """
        Добавляет одну детекцию из кадра в текущую
        пространственную группу.
        """

        if not self._valid_detection(
            letter,
            confidence,
            latitude,
            longitude,
        ):
            return

        now = time.monotonic()

        detection = {
            "letter": str(letter),
            "confidence": float(confidence),
            "latitude": float(latitude),
            "longitude": float(longitude),
        }

        if self.current_track is None:
            self.current_track = self._new_track(detection, now)
            return

        if self._belongs_to_current_track(detection, now):
            self._add_to_current_track(detection, now)
            return

        self.flush_current_track()
        self.current_track = self._new_track(detection, now)

    def update(self):
        """
        Вызывать один раз за проход цикла кадров.
        Если буква исчезла из кадров — закрывает группу,
        выбирает победителя по sum(confidence ** 2) и
        сохраняет одну строку в CSV.
        """

        if self.current_track is None:
            return

        elapsed = (
            time.monotonic() - self.current_track["last_seen"]
        )

        if elapsed >= self.track_timeout_s:
            self.flush_current_track()

    def close(self):
        """Вызывать перед остановкой Python-скрипта."""

        self.flush_current_track()

    def _new_track(self, detection, now):
        track = {
            "latitude": detection["latitude"],
            "longitude": detection["longitude"],
            "last_seen": now,
            "frames_count": 0,
            "letter_scores": {},
            "letter_best": {},
        }

        self.current_track = track
        self._add_to_current_track(detection, now)

        return track

    def _belongs_to_current_track(self, detection, now):
        track = self.current_track

        if now - track["last_seen"] > self.track_timeout_s:
            return False

        dist = distance_m(
            detection["latitude"],
            detection["longitude"],
            track["latitude"],
            track["longitude"],
        )

        return dist <= self.different_letter_distance_m

    def _add_to_current_track(self, detection, now):
        track = self.current_track
        letter = detection["letter"]
        confidence = detection["confidence"]

        track["last_seen"] = now
        track["frames_count"] += 1

        # Главная часть: накопление confidence².
        track["letter_scores"][letter] = (
            track["letter_scores"].get(letter, 0.0)
            + confidence ** 2
        )

        # Для координат победившей буквы оставляем детекцию
        # с максимальным confidence.
        best_for_letter = track["letter_best"].get(letter)

        if (
            best_for_letter is None
            or confidence > best_for_letter["confidence"]
        ):
            track["letter_best"][letter] = detection

    def _winner(self):
        """
        Возвращает букву с максимальной суммой confidence²
        и координаты её самого уверенного кадра.
        """

        track = self.current_track

        winner_letter = max(
            track["letter_scores"],
            key=track["letter_scores"].get,
        )

        best_detection = track["letter_best"][winner_letter]

        return (
            winner_letter,
            best_detection["latitude"],
            best_detection["longitude"],
            track["letter_scores"][winner_letter],
        )

    def _already_saved(self, letter, latitude, longitude):
        """
        Проверяет, не была ли эта же буква уже сохранена
        ближе чем на 25 метров.
        """

        for saved in self.saved_letters:
            if saved["letter"] != letter:
                continue

            dist = distance_m(
                latitude,
                longitude,
                saved["latitude"],
                saved["longitude"],
            )

            if dist <= self.different_letter_distance_m:
                return True

        return False

    def flush_current_track(self):
        """
        Закрывает текущую группу и записывает одну строку:
        letter;latitude;longitude.
        """

        if self.current_track is None:
            return

        track = self.current_track

        if track["frames_count"] < self.min_frames:
            self.current_track = None
            return

        letter, latitude, longitude, score = self._winner()

        if self._already_saved(letter, latitude, longitude):
            print(
                f"CSV: {letter} уже сохранена в пределах "
                f"{self.different_letter_distance_m:.0f} м — пропуск",
                flush=True,
            )
            self.current_track = None
            return

        with self.csv_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow([
                letter,
                f"{latitude:.7f}",
                f"{longitude:.7f}",
            ])

        self.saved_letters.append(
            {
                "letter": letter,
                "latitude": latitude,
                "longitude": longitude,
            }
        )

        print(
            f"CSV: сохранена {letter}; "
            f"score={score:.3f}; "
            f"coord={latitude:.7f}, {longitude:.7f}",
            flush=True,
        )

        self.current_track = None

    def _valid_detection(
        self,
        letter,
        confidence,
        latitude,
        longitude,
    ):
        if letter is None:
            return False

        if confidence is None:
            return False

        confidence = float(confidence)

        if confidence < self.min_confidence:
            return False


        return True