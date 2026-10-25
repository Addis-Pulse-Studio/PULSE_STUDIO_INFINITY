"""The one durable-write path every crash-safe record goes through.

The manifest has had this behaviour since §7.4; these tests pin it on the shared
helper, because Pulse Infinity's session, memory, desk and journal now rely on it
too, and a regression here would corrupt all five at once.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from comfyui_pulse_studio import durable


class TestDurableWrites(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_write_lands_whole_and_leaves_no_temp_file(self):
        path = os.path.join(self.directory, "state.json")
        durable.write_json_durably(path, {"beat": 3, "line": "ሰላም"})
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh), {"beat": 3, "line": "ሰላም"})
        self.assertEqual(sorted(os.listdir(self.directory)), ["state.json"])

    def test_dialogue_is_written_readably_not_escaped(self):
        """A person opens these files to see what the director decided."""
        path = os.path.join(self.directory, "plan.json")
        durable.write_json_durably(path, {"text": "ጊዜው ቆሟል"})
        with open(path, encoding="utf-8") as fh:
            self.assertIn("ጊዜው ቆሟል", fh.read())

    def test_a_failed_replace_keeps_the_old_file_and_removes_the_temp(self):
        path = os.path.join(self.directory, "state.json")
        durable.write_json_durably(path, {"v": 1})
        with mock.patch.object(durable.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                durable.write_json_durably(path, {"v": 2})
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh), {"v": 1})
        self.assertEqual(sorted(os.listdir(self.directory)), ["state.json"])

    def test_the_target_folder_is_created(self):
        path = os.path.join(self.directory, "desk", "inbox", "b0001-open-a1.json")
        durable.write_json_durably(path, {})
        self.assertTrue(os.path.isfile(path))

    def test_the_temp_file_is_made_beside_the_target(self):
        """os.replace is atomic only within one filesystem."""
        made = []
        real = tempfile.mkstemp

        def spy(*args, **kwargs):
            made.append(kwargs.get("dir"))
            return real(*args, **kwargs)

        path = os.path.join(self.directory, "state.json")
        with mock.patch.object(durable.tempfile, "mkstemp", side_effect=spy):
            durable.write_text_durably(path, "x")
        self.assertEqual(made, [os.path.dirname(os.path.abspath(path))])


if __name__ == "__main__":
    unittest.main()
