#!/usr/bin/env python3
"""Replay one persisted Stage2-R-v2 failure batch on CPU or NPU."""
from __future__ import annotations

import argparse,json,sys,traceback
from contextlib import nullcontext
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent;V1=HERE.parent/"stage2_r_recurrent_critic_pretraining";VENDOR=HERE.parent/"third_party"/"pomdp_baselines"
for path in (HERE,V1,VENDOR):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from stage2_r_critic import make_pair,set_device  # noqa: E402
from stage2_r_v2_critic import forward_values  # noqa: E402


def read(path):
    with open(path,encoding="utf-8") as handle:return json.load(handle)
def device_of(name):
    if name.startswith("npu"):
        import torch_npu  # noqa
        if not torch.npu.is_available():raise RuntimeError("Ascend NPU unavailable")
        torch.npu.set_device(name)
    return torch.device(name)
def move(batch,device):return {key:value.to(device=device,dtype=torch.float32) for key,value in batch.items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--failure-dir",required=True);parser.add_argument("--config",default=str(HERE/"stage2_r_v2_config.json"));parser.add_argument("--device",default="cpu");parser.add_argument("--detect-anomaly",action="store_true");args=parser.parse_args()
    failure=Path(args.failure_dir);config=read(args.config);device=device_of(args.device);set_device(device)
    critic,target=make_pair(config,device);critic.load_state_dict(torch.load(failure/"critic_before_failure.pth",map_location=device));target.load_state_dict(torch.load(failure/"target_critic_before_failure.pth",map_location=device));target.requires_grad_(False)
    batch=move(torch.load(failure/"batch.pt",map_location="cpu"),device);critic.train();critic.zero_grad(set_to_none=True)
    result={"failure_dir":str(failure.resolve()),"device":str(device),"detect_anomaly":args.detect_anomaly,"forward_finite":False,"loss_finite":False,"gradient_finite":False,"bad_gradient_parameter_names":[],"bad_gradient_count":0,"first_detected_nonfinite_operation":None}
    try:
        context=torch.autograd.detect_anomaly(check_nan=True) if args.detect_anomaly else nullcontext()
        with context:
            q1,q2,target_q,bellman,hidden,state_norms=forward_values(critic,target,batch,config["gamma"])
            result["forward_finite"]=bool(all(torch.isfinite(value).all() for value in (q1,q2,target_q,bellman,hidden)))
            mask=batch["mask"];valid=torch.clamp(mask.sum(),min=1.0);loss=((((q1-bellman)**2)+((q2-bellman)**2))*mask).sum()/valid
            result["loss"]=float(loss.detach().item());result["loss_finite"]=bool(torch.isfinite(loss));result.update(state_norms);loss.backward()
        for name,parameter in critic.named_parameters():
            if parameter.grad is None:continue
            count=int((~torch.isfinite(parameter.grad)).sum().item())
            if count:result["bad_gradient_parameter_names"].append(name);result["bad_gradient_count"]+=count
        result["gradient_finite"]=result["bad_gradient_count"]==0
    except Exception as error:
        result["first_detected_nonfinite_operation"]=f"{type(error).__name__}: {error}";result["traceback"]=traceback.format_exc()
    output=failure/f"debug_replay_{str(device).replace(':','_')}{'_anomaly' if args.detect_anomaly else ''}.json";output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False));print("Saved:",output)
    if not result["forward_finite"] or not result["loss_finite"] or not result["gradient_finite"]:raise SystemExit(1)
if __name__=="__main__":main()
