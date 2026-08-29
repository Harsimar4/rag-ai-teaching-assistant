"""
Unit tests for the data pipeline utilities:
  - video_to_mp3.py: filename parsing, extension filtering
  - mp3_to_json.py: filename parsing, segment conversion
  - merge_chunks.py: chunk merging logic
  - preprocess_json.py / preprocess_new_json.py: batch helpers, validation
"""

import pytest


# ---------------------------------------------------------------------------
# video_to_mp3: parse_video_filename
# ---------------------------------------------------------------------------

class TestParseVideoFilename:
    def test_standard_filename(self):
        from video_to_mp3 import parse_video_filename

        number, title = parse_video_filename("03 - Basic Structure of HTML.webm")
        assert number == "03"
        assert title == "Basic Structure of HTML.webm"

    def test_filename_with_special_separator(self):
        from video_to_mp3 import parse_video_filename

        number, title = parse_video_filename("10 - CSS Selectors｜Extra Info.mp4")
        assert number == "10"
        assert title == "CSS Selectors"

    def test_single_digit_number(self):
        from video_to_mp3 import parse_video_filename

        number, title = parse_video_filename("1 - Intro.mkv")
        assert number == "1"


# ---------------------------------------------------------------------------
# video_to_mp3: VIDEO_EXTENSIONS filter
# ---------------------------------------------------------------------------

class TestVideoExtensions:
    def test_supported_extensions(self):
        from video_to_mp3 import VIDEO_EXTENSIONS

        assert ".mp4" in VIDEO_EXTENSIONS
        assert ".webm" in VIDEO_EXTENSIONS
        assert ".mkv" in VIDEO_EXTENSIONS
        assert ".avi" in VIDEO_EXTENSIONS
        assert ".mov" in VIDEO_EXTENSIONS

    def test_unsupported_extensions(self):
        from video_to_mp3 import VIDEO_EXTENSIONS

        assert ".txt" not in VIDEO_EXTENSIONS
        assert ".mp3" not in VIDEO_EXTENSIONS
        assert ".json" not in VIDEO_EXTENSIONS
        assert ".gitkeep" not in VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# mp3_to_json: parse_audio_filename
# ---------------------------------------------------------------------------

class TestParseAudioFilename:
    def test_standard_filename(self):
        from mp3_to_json import parse_audio_filename

        number, title = parse_audio_filename("03_Basic Structure.mp3")
        assert number == "03"
        assert title == "Basic Structure"

    def test_title_with_underscores(self):
        from mp3_to_json import parse_audio_filename

        number, title = parse_audio_filename("10_CSS_Selectors_Advanced.mp3")
        assert number == "10"
        assert title == "CSS_Selectors_Advanced"


# ---------------------------------------------------------------------------
# mp3_to_json: segments_to_chunks
# ---------------------------------------------------------------------------

class TestSegmentsToChunks:
    def test_converts_segments(self):
        from mp3_to_json import segments_to_chunks

        segments = [
            {"start": 0.0, "end": 5.5, "text": "Hello world"},
            {"start": 5.5, "end": 12.0, "text": "Second segment"},
        ]

        chunks = segments_to_chunks(segments, "03", "My Title")
        assert len(chunks) == 2
        assert chunks[0]["number"] == "03"
        assert chunks[0]["title"] == "My Title"
        assert chunks[0]["start"] == 0.0
        assert chunks[0]["text"] == "Hello world"

    def test_empty_segments(self):
        from mp3_to_json import segments_to_chunks

        chunks = segments_to_chunks([], "1", "Title")
        assert chunks == []


# ---------------------------------------------------------------------------
# merge_chunks: merge_chunk_group, merge_file_chunks
# ---------------------------------------------------------------------------

class TestMergeChunkGroup:
    def test_merges_text(self):
        from merge_chunks import merge_chunk_group

        group = [
            {"number": "1", "title": "T", "start": 0.0, "end": 5.0, "text": "hello"},
            {"number": "1", "title": "T", "start": 5.0, "end": 10.0, "text": "world"},
        ]
        merged = merge_chunk_group(group)
        assert merged["text"] == "hello world"
        assert merged["start"] == 0.0
        assert merged["end"] == 10.0
        assert merged["number"] == "1"

    def test_single_chunk(self):
        from merge_chunks import merge_chunk_group

        group = [{"number": "1", "title": "T", "start": 0.0, "end": 5.0, "text": "only"}]
        merged = merge_chunk_group(group)
        assert merged["text"] == "only"


class TestMergeFileChunks:
    def test_groups_correctly(self):
        from merge_chunks import merge_file_chunks

        chunks = [
            {"number": "1", "title": "T", "start": float(i), "end": float(i + 1), "text": f"s{i}"}
            for i in range(7)
        ]
        # group_size = 3 should produce 3 groups (3, 3, 1)
        merged = merge_file_chunks(chunks, group_size=3)
        assert len(merged) == 3
        assert merged[0]["text"] == "s0 s1 s2"
        assert merged[2]["text"] == "s6"

    def test_exact_division(self):
        from merge_chunks import merge_file_chunks

        chunks = [
            {"number": "1", "title": "T", "start": float(i), "end": float(i + 1), "text": f"s{i}"}
            for i in range(6)
        ]
        merged = merge_file_chunks(chunks, group_size=3)
        assert len(merged) == 2

    def test_empty_chunks(self):
        from merge_chunks import merge_file_chunks

        merged = merge_file_chunks([], group_size=3)
        assert merged == []


# ---------------------------------------------------------------------------
# preprocess_json: batch_items, load_valid_chunks
# ---------------------------------------------------------------------------

class TestBatchItems:
    def test_even_batches(self):
        from preprocess_json import batch_items

        batches = list(batch_items([1, 2, 3, 4], 2))
        assert batches == [(0, [1, 2]), (2, [3, 4])]

    def test_uneven_batches(self):
        from preprocess_json import batch_items

        batches = list(batch_items([1, 2, 3, 4, 5], 2))
        assert len(batches) == 3
        assert batches[-1] == (4, [5])

    def test_empty_list(self):
        from preprocess_json import batch_items

        batches = list(batch_items([], 2))
        assert batches == []

    def test_batch_size_larger_than_list(self):
        from preprocess_json import batch_items

        batches = list(batch_items([1, 2], 10))
        assert batches == [(0, [1, 2])]


class TestLoadValidChunks:
    def test_filters_empty_text(self):
        from preprocess_json import load_valid_chunks

        content = {
            "chunks": [
                {"text": "valid chunk"},
                {"text": "   "},
                {"text": ""},
                {"text": "another valid"},
            ]
        }
        valid = load_valid_chunks(content)
        assert len(valid) == 2

    def test_filters_non_dict(self):
        from preprocess_json import load_valid_chunks

        content = {"chunks": ["not a dict", {"text": "valid"}]}
        valid = load_valid_chunks(content)
        assert len(valid) == 1

    def test_no_chunks_key(self):
        from preprocess_json import load_valid_chunks

        content = {"data": []}
        valid = load_valid_chunks(content)
        assert valid == []

    def test_all_valid(self):
        from preprocess_json import load_valid_chunks

        content = {"chunks": [{"text": "a"}, {"text": "b"}]}
        valid = load_valid_chunks(content)
        assert len(valid) == 2
