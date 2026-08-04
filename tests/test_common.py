import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ID_TO_LABEL, LABEL_MAP, parse_int_list

class CommonTests(unittest.TestCase):
 def test_label_mapping_is_bidirectional(self):
    self.assertEqual(len(LABEL_MAP), 4)
    self.assertEqual({ID_TO_LABEL[value] for value in LABEL_MAP.values()}, set(LABEL_MAP))

 def test_parse_int_list(self):
    self.assertEqual(parse_int_list(["42", "52", "62"]), [42, 52, 62])
