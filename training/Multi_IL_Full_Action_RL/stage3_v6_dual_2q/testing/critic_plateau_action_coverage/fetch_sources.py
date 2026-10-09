import urllib.request,json,base64,concurrent.futures
from pathlib import Path
D=Path(__file__).parent/'pirlnav_sources'; commit=json.load(urllib.request.urlopen(urllib.request.Request('https://api.github.com/repos/Ram81/pirlnav/commits/main',headers={'User-Agent':'audit'}),timeout=30))['sha']
def fetch(f):
 u='https://api.github.com/repos/Ram81/pirlnav/contents/pirlnav/'+f+'?ref='+commit
 try:
  req=urllib.request.Request(u,headers={'User-Agent':'audit','Accept':'application/vnd.github+json'}); x=json.load(urllib.request.urlopen(req,timeout=40)); data=base64.b64decode(x['content']); (D/f.replace('/','__')).write_bytes(data); print(f,len(data),flush=True)
 except Exception as e: print(f,repr(e),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as e: list(e.map(fetch,['ppo_trainer.py','algos/ppo.py','utils/lr_scheduler.py','policy/policy.py','config.py']))
(D/'commit.txt').write_text(commit)
