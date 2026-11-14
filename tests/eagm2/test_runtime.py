import unittest,tempfile
from eagm2.runtime import check_seed,stream_seed,new_run
class RuntimeTests(unittest.TestCase):
 def test_confirm_locked(self):
  for task,seed in [('tracking',200101),('contact',201101),('vision',202101),('tracking',0)]:
   with self.assertRaises(PermissionError):check_seed(task,seed)
 def test_development_only(self):
  check_seed('tracking',200001)
  with self.assertRaises(PermissionError):check_seed('vision',200001)
 def test_streams(self):
  self.assertEqual(stream_seed(200001,'train'),stream_seed(200001,'train'))
  self.assertNotEqual(stream_seed(200001,'train'),stream_seed(200001,'eval'))
if __name__=='__main__':unittest.main()
