"""
Tests for src.pipeline.process_game

Coverage targets:
- load_session_from_local: happy path (warped/ subfolder), fallback to root dir,
  legacy filename format, missing game_info.json, no images, unreadable images
- load_session_from_s3: happy path, warped/ fallback to root prefix, empty bucket
- process_game_session: no source provided, no images, result from game_info,
  custom vs pretrained classifier dispatch, full pipeline with mocked deps
"""

import json
import os
import re

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_board_image():
    """Return a minimal 400x400 BGR numpy array (solid grey)."""
    return np.full((400, 400, 3), 128, dtype=np.uint8)


def _write_jpg(path, img=None):
    if img is None:
        img = _make_board_image()
    cv2.imwrite(str(path), img)


def _game_info(extra=None):
    base = {
        "board_grid": {
            "x_lines": [i * 50 for i in range(9)],
            "y_lines": [i * 50 for i in range(9)],
        },
        "rotation_angle": 0,
    }
    if extra:
        base.update(extra)
    return base


# ---------------------------------------------------------------------------
# load_session_from_local
# ---------------------------------------------------------------------------

class TestLoadSessionFromLocal:
    def test_happy_path_warped_subfolder(self, tmp_path):
        """Images in warped/ are loaded and sorted by trailing frame number."""
        from src.pipeline.process_game import load_session_from_local

        (tmp_path / "warped").mkdir()
        info = _game_info()
        (tmp_path / "game_info.json").write_text(json.dumps(info))
        _write_jpg(tmp_path / "warped" / "002.jpg")
        _write_jpg(tmp_path / "warped" / "000.jpg")
        _write_jpg(tmp_path / "warped" / "001.jpg")

        result = load_session_from_local(str(tmp_path))

        assert result["game_info"] == info
        assert len(result["images"]) == 3
        # All are valid numpy arrays
        for img in result["images"]:
            assert isinstance(img, np.ndarray)

    def test_images_sorted_by_frame_number(self, tmp_path):
        """Frame order follows trailing numeric suffix, not lexicographic order."""
        from src.pipeline.process_game import load_session_from_local

        (tmp_path / "game_info.json").write_text(json.dumps(_game_info()))
        # Create images with distinct pixel values so we can verify order
        for idx in [9, 1, 5]:
            img = np.full((400, 400, 3), idx * 10, dtype=np.uint8)
            _write_jpg(tmp_path / f"{idx:03d}.jpg", img)

        result = load_session_from_local(str(tmp_path))
        # Pixel means should be 10, 50, 90 (frames 001, 005, 009)
        means = [img.mean() for img in result["images"]]
        assert means == sorted(means)

    def test_legacy_filename_format(self, tmp_path):
        """Filenames like SESSION_ABC_000.jpg are accepted and sorted by suffix."""
        from src.pipeline.process_game import load_session_from_local

        (tmp_path / "game_info.json").write_text(json.dumps(_game_info()))
        _write_jpg(tmp_path / "SESSION_XYZ_001.jpg")
        _write_jpg(tmp_path / "SESSION_XYZ_000.jpg")

        result = load_session_from_local(str(tmp_path))
        assert len(result["images"]) == 2

    def test_falls_back_to_root_when_no_warped_dir(self, tmp_path):
        """When warped/ does not exist, images are loaded from session root."""
        from src.pipeline.process_game import load_session_from_local

        (tmp_path / "game_info.json").write_text(json.dumps(_game_info()))
        _write_jpg(tmp_path / "000.jpg")

        result = load_session_from_local(str(tmp_path))
        assert len(result["images"]) == 1

    def test_missing_game_info_raises(self, tmp_path):
        """FileNotFoundError is raised when game_info.json is absent."""
        from src.pipeline.process_game import load_session_from_local

        with pytest.raises(FileNotFoundError, match="game_info.json"):
            load_session_from_local(str(tmp_path))

    def test_no_images_returns_empty_list(self, tmp_path):
        """A session with no image files returns an empty images list."""
        from src.pipeline.process_game import load_session_from_local

        (tmp_path / "game_info.json").write_text(json.dumps(_game_info()))
        result = load_session_from_local(str(tmp_path))
        assert result["images"] == []

    def test_non_image_files_ignored(self, tmp_path):
        """Non-image files (txt, json) in the session dir are skipped."""
        from src.pipeline.process_game import load_session_from_local

        (tmp_path / "game_info.json").write_text(json.dumps(_game_info()))
        (tmp_path / "notes.txt").write_text("ignore me")
        _write_jpg(tmp_path / "000.jpg")

        result = load_session_from_local(str(tmp_path))
        assert len(result["images"]) == 1

    def test_game_info_fields_preserved(self, tmp_path):
        """Arbitrary game_info fields are returned verbatim."""
        from src.pipeline.process_game import load_session_from_local

        info = _game_info({"white": "Alice", "black": "Bob", "result": "1-0"})
        (tmp_path / "game_info.json").write_text(json.dumps(info))

        result = load_session_from_local(str(tmp_path))
        assert result["game_info"]["white"] == "Alice"
        assert result["game_info"]["result"] == "1-0"


# ---------------------------------------------------------------------------
# load_session_from_s3
# ---------------------------------------------------------------------------

class TestLoadSessionFromS3:
    """All S3 tests mock boto3 to avoid network calls."""

    def _make_s3_client(self, mocker, *, game_info, image_keys=None, use_root_fallback=False):
        """Return a mock boto3 client pre-configured with the given data."""
        import io
        import unittest.mock as mock

        info_bytes = json.dumps(game_info).encode()
        game_id = "test-game-123"
        bucket = "test-bucket"

        warped_prefix = f"sessions/{game_id}/warped/"
        root_prefix = f"sessions/{game_id}/"

        def get_object(Bucket, Key):
            if Key == f"{root_prefix}game_info.json":
                return {"Body": io.BytesIO(info_bytes)}
            # Return a tiny 400x400 grey JPEG for image keys
            img = _make_board_image()
            ok, buf = cv2.imencode(".jpg", img)
            return {"Body": io.BytesIO(buf.tobytes())}

        # Build paginator pages
        warped_contents = []
        root_contents = []
        if image_keys:
            for k in image_keys:
                if "warped" in k:
                    warped_contents.append({"Key": k})
                else:
                    root_contents.append({"Key": k})

        def paginate(Bucket, Prefix):
            if Prefix == warped_prefix:
                if use_root_fallback:
                    return [{"Contents": []}]
                return [{"Contents": warped_contents}]
            return [{"Contents": root_contents}]

        mock_paginator = mock.MagicMock()
        mock_paginator.paginate.side_effect = paginate

        mock_client = mock.MagicMock()
        mock_client.get_object.side_effect = get_object
        mock_client.get_paginator.return_value = mock_paginator

        return mock_client, game_id, bucket

    def test_happy_path_warped_prefix(self, mocker):
        """Images under sessions/{id}/warped/ are loaded and returned."""
        from src.pipeline.process_game import load_session_from_s3
        import unittest.mock as mock

        info = _game_info()
        game_id = "test-game-123"
        image_keys = [
            f"sessions/{game_id}/warped/000.jpg",
            f"sessions/{game_id}/warped/001.jpg",
        ]
        mock_client, game_id, bucket = self._make_s3_client(
            mocker, game_info=info, image_keys=image_keys
        )

        mocker.patch("boto3.client", return_value=mock_client)
        mocker.patch("dotenv.load_dotenv")

        result = load_session_from_s3(game_id, bucket=bucket)
        assert result["game_info"] == info
        assert len(result["images"]) == 2

    def test_fallback_to_root_prefix_when_warped_empty(self, mocker):
        """When warped/ prefix is empty, root prefix images are used."""
        from src.pipeline.process_game import load_session_from_s3

        info = _game_info()
        game_id = "test-game-123"
        image_keys = [f"sessions/{game_id}/000.jpg"]
        mock_client, game_id, bucket = self._make_s3_client(
            mocker, game_info=info, image_keys=image_keys, use_root_fallback=True
        )

        mocker.patch("boto3.client", return_value=mock_client)
        mocker.patch("dotenv.load_dotenv")

        result = load_session_from_s3(game_id, bucket=bucket)
        assert len(result["images"]) == 1

    def test_empty_bucket_returns_empty_images(self, mocker):
        """No image keys → images list is empty."""
        from src.pipeline.process_game import load_session_from_s3

        info = _game_info()
        game_id = "test-game-123"
        mock_client, game_id, bucket = self._make_s3_client(
            mocker, game_info=info, image_keys=[], use_root_fallback=True
        )

        mocker.patch("boto3.client", return_value=mock_client)
        mocker.patch("dotenv.load_dotenv")

        result = load_session_from_s3(game_id, bucket=bucket)
        assert result["images"] == []

    def test_images_sorted_by_frame_number(self, mocker):
        """Keys are ordered by trailing numeric suffix regardless of S3 order."""
        from src.pipeline.process_game import load_session_from_s3
        import io
        import unittest.mock as mock

        game_id = "test-game-123"
        info = _game_info()

        # S3 returns keys out of order
        keys_in_s3_order = [
            f"sessions/{game_id}/warped/002.jpg",
            f"sessions/{game_id}/warped/000.jpg",
            f"sessions/{game_id}/warped/001.jpg",
        ]

        mock_paginator = mock.MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": k} for k in keys_in_s3_order]}
        ]

        call_order = []

        def get_object(Bucket, Key):
            if "game_info" in Key:
                return {"Body": io.BytesIO(json.dumps(info).encode())}
            frame = int(re.search(r"(\d+)\.jpg", Key).group(1))
            call_order.append(frame)
            img = np.full((400, 400, 3), frame * 10, dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", img)
            return {"Body": io.BytesIO(buf.tobytes())}

        mock_client = mock.MagicMock()
        mock_client.get_object.side_effect = get_object
        mock_client.get_paginator.return_value = mock_paginator

        mocker.patch("boto3.client", return_value=mock_client)
        mocker.patch("dotenv.load_dotenv")

        load_session_from_s3(game_id, bucket="test-bucket")
        assert call_order == sorted(call_order)


# ---------------------------------------------------------------------------
# process_game_session
# ---------------------------------------------------------------------------

class TestProcessGameSession:
    """Full pipeline tests — classifier and S3 are mocked."""

    def _mock_classifier(self, mocker, piece="empty"):
        """Return a mock ChessPieceClassifier whose predict_board returns all-'piece'."""
        import unittest.mock as mock

        clf = mock.MagicMock()
        predictions = {
            sq: (piece, 0.99)
            for sq in [
                f"{f}{r}"
                for f in "abcdefgh"
                for r in "12345678"
            ]
        }
        clf.predict_board.return_value = predictions
        mock_cls = mock.MagicMock(return_value=clf)
        mocker.patch("src.pipeline.process_game.ChessPieceClassifier", mock_cls)
        mocker.patch("src.pipeline.process_game.crop_squares_from_grid",
                     return_value={sq: np.zeros((50, 50, 3), dtype=np.uint8)
                                   for sq in predictions})
        return clf

    def test_raises_when_no_source_given(self):
        """ValueError is raised when neither game_id nor local_dir is provided."""
        from src.pipeline.process_game import process_game_session

        with pytest.raises(ValueError, match="game_id or local_dir"):
            process_game_session()

    def test_raises_when_session_has_no_images(self, tmp_path, mocker):
        """ValueError is raised when the session directory contains no images."""
        from src.pipeline.process_game import process_game_session

        (tmp_path / "game_info.json").write_text(json.dumps(_game_info()))
        self._mock_classifier(mocker)

        with pytest.raises(ValueError, match="No images"):
            process_game_session(local_dir=str(tmp_path))

    def test_result_taken_from_game_info(self, tmp_path, mocker):
        """When result='*' (default) and game_info has a result, it is used in PGN."""
        from src.pipeline.process_game import process_game_session

        info = _game_info({"result": "1-0"})
        (tmp_path / "game_info.json").write_text(json.dumps(info))
        _write_jpg(tmp_path / "000.jpg")
        _write_jpg(tmp_path / "001.jpg")
        self._mock_classifier(mocker)

        out = process_game_session(local_dir=str(tmp_path))
        assert "1-0" in out["pgn"]

    def test_explicit_result_overrides_game_info(self, tmp_path, mocker):
        """An explicit result argument takes precedence over game_info['result']."""
        from src.pipeline.process_game import process_game_session

        info = _game_info({"result": "1-0"})
        (tmp_path / "game_info.json").write_text(json.dumps(info))
        _write_jpg(tmp_path / "000.jpg")
        self._mock_classifier(mocker)

        out = process_game_session(local_dir=str(tmp_path), result="0-1")
        assert "0-1" in out["pgn"]

    def test_returns_expected_keys(self, tmp_path, mocker):
        """Return dict always contains pgn, moves, fen_sequence, errors, skipped."""
        from src.pipeline.process_game import process_game_session

        (tmp_path / "game_info.json").write_text(json.dumps(_game_info()))
        _write_jpg(tmp_path / "000.jpg")
        self._mock_classifier(mocker)

        out = process_game_session(local_dir=str(tmp_path))
        for key in ("pgn", "moves", "fen_sequence", "errors", "skipped"):
            assert key in out

    def test_fen_sequence_length_matches_images(self, tmp_path, mocker):
        """One FEN is generated per input image."""
        from src.pipeline.process_game import process_game_session

        (tmp_path / "game_info.json").write_text(json.dumps(_game_info()))
        for i in range(5):
            _write_jpg(tmp_path / f"{i:03d}.jpg")
        self._mock_classifier(mocker)

        out = process_game_session(local_dir=str(tmp_path))
        assert len(out["fen_sequence"]) == 5

    def test_pretrained_classifier_instantiated(self, tmp_path, mocker):
        """When classifier='pretrained', PretrainedBoardClassifier is used."""
        import unittest.mock as mock
        from src.pipeline.process_game import process_game_session

        (tmp_path / "game_info.json").write_text(json.dumps(_game_info()))
        _write_jpg(tmp_path / "000.jpg")

        clf = mock.MagicMock()
        predictions = {
            f"{f}{r}": ("empty", 0.99)
            for f in "abcdefgh"
            for r in "12345678"
        }
        clf.predict_board.return_value = predictions
        pretrained_cls = mock.MagicMock(return_value=clf)
        mocker.patch("src.pipeline.process_game.PretrainedBoardClassifier", pretrained_cls)
        mocker.patch("src.pipeline.process_game.ChessPieceClassifier")
        mocker.patch("src.pipeline.process_game.crop_squares_from_grid",
                     return_value={sq: np.zeros((50, 50, 3), dtype=np.uint8)
                                   for sq in predictions})

        process_game_session(local_dir=str(tmp_path), classifier="pretrained")

        pretrained_cls.assert_called_once()

    def test_fallback_grid_used_when_missing(self, tmp_path, mocker):
        """If board_grid is absent from game_info, a uniform 50px grid is used."""
        import unittest.mock as mock
        from src.pipeline.process_game import process_game_session

        info = {"rotation_angle": 0}  # no board_grid key
        (tmp_path / "game_info.json").write_text(json.dumps(info))
        _write_jpg(tmp_path / "000.jpg")

        captured_grids = []
        real_clf = self._mock_classifier(mocker)

        original_crop = __import__(
            "src.preprocessing.process_board", fromlist=["crop_squares_from_grid"]
        ).crop_squares_from_grid

        def recording_crop(img, grid, rotation):
            captured_grids.append(grid)
            return {
                f"{f}{r}": np.zeros((50, 50, 3), dtype=np.uint8)
                for f in "abcdefgh"
                for r in "12345678"
            }

        mocker.patch("src.pipeline.process_game.crop_squares_from_grid", recording_crop)

        process_game_session(local_dir=str(tmp_path))

        assert captured_grids, "crop_squares_from_grid was never called"
        grid_used = captured_grids[0]
        assert "x_lines" in grid_used
        assert len(grid_used["x_lines"]) == 9

    def test_s3_path_delegates_to_load_session_from_s3(self, mocker):
        """When game_id is given (no local_dir), load_session_from_s3 is called."""
        import unittest.mock as mock
        from src.pipeline.process_game import process_game_session

        fake_session = {
            "game_info": _game_info(),
            "images": [_make_board_image()],
        }
        mock_load = mocker.patch(
            "src.pipeline.process_game.load_session_from_s3",
            return_value=fake_session,
        )
        self._mock_classifier(mocker)

        process_game_session(game_id="abc-123", s3_bucket="my-bucket")

        mock_load.assert_called_once_with("abc-123", "my-bucket")
