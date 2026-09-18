import tempfile, unittest
from pathlib import Path
import h5py, numpy as np
from stage3_v5_replay import CANONICAL_KEYS, Stage1OfflineSequenceReplay, BalancedOfflineDemonstrations

def make_file(path):
 with h5py.File(path,"w") as f:
  f.attrs["canonical_observation_keys"]=__import__("json").dumps(CANONICAL_KEYS); eps=f.create_group("episodes")
  for i,(term,trunc) in enumerate(((1,0),(0,1))):
   g=eps.create_group(str(i)); n=12; g.create_dataset("actions",data=np.zeros((n,14),np.float32)); g.create_dataset("rewards",data=np.zeros(n)); g.create_dataset("dones",data=np.r_[np.zeros(n-1),1]); g.create_dataset("terminated",data=np.r_[np.zeros(n-1),term]); g.create_dataset("truncated",data=np.r_[np.zeros(n-1),trunc])
   dims=(3,4,2,3,4,2,41)
   for parent in (g.create_group("obs"),g.create_group("next_obs")):
    for key,width in zip(CANONICAL_KEYS,dims): parent.create_dataset(key,data=np.zeros((n,width),np.float32))

class TestReplay(unittest.TestCase):
 def test_native_loader_and_rotation(self):
  with tempfile.TemporaryDirectory() as d:
   paths=[str(Path(d)/f"{i}.h5") for i in range(3)]
   for p in paths: make_file(p)
   one=Stage1OfflineSequenceReplay(paths[0],"rnn"); self.assertTrue(one.episodes[0]["terminated"][-1]); self.assertTrue(one.episodes[1]["truncated"][-1])
   multi=BalancedOfflineDemonstrations(paths,1); totals=np.zeros(3,int)
   for _ in range(3):
    ids=multi.sample_sequences(128,10)["source_id"]; self.assertEqual(len(ids),128); totals += [(ids==i).sum() for i in range(3)]
   self.assertTrue(np.array_equal(totals,[128,128,128]))
