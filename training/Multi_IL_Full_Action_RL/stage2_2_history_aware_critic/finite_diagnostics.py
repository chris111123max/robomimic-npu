"""Finite guards and compact numerical diagnostics for Stage2.2."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
import torch

def tensor_stats(value):
    tensor=torch.as_tensor(value).detach();finite=torch.isfinite(tensor);result={"shape":list(tensor.shape),"isfinite":bool(finite.all()),"non_finite_count":int((~finite).sum())}
    clean=tensor[finite].float()
    if clean.numel():result.update(min=float(clean.min()),max=float(clean.max()),mean=float(clean.mean()),std=float(clean.std(unbiased=False)),abs_max=float(clean.abs().max()))
    else:result.update(min=None,max=None,mean=None,std=None,abs_max=None)
    return result

def non_finite_names(named_values):
    return [name for name,value in named_values if value is not None and not torch.isfinite(value).all()]

def parameter_statistics(model):
    out={}
    for twin in ("q1","q2"):
        values=[p.detach().reshape(-1) for p in getattr(model,twin).parameters()]
        flat=torch.cat(values) if values else torch.empty(0)
        out[f"{twin}_max_abs_parameter"]=float(flat.abs().max()) if flat.numel() and torch.isfinite(flat).all() else None
        out[f"{twin}_parameters_finite"]=bool(torch.isfinite(flat).all())
    return out

def gradient_statistics(model):
    result={};bad=[]
    for twin in ("q1","q2"):
        module=getattr(model,twin);groups={"total":list(module.named_parameters())}
        for name in ("token_encoder","lstm","q_head","feature_encoder"):
            if hasattr(module,name):groups[name]=list(getattr(module,name).named_parameters(prefix=name))
        for group,items in groups.items():
            grads=[p.grad.detach().reshape(-1) for _,p in items if p.grad is not None]
            flat=torch.cat(grads) if grads else torch.empty(0,device=next(module.parameters()).device)
            prefix=f"{twin}_{group}" if group!="total" else twin
            result[f"{prefix}_grad_norm"]=float(torch.linalg.vector_norm(flat.float())) if flat.numel() else 0.0
            result[f"{prefix}_max_abs_grad"]=float(flat.abs().max()) if flat.numel() and torch.isfinite(flat).all() else None
            for name,p in items:
                if p.grad is not None and not torch.isfinite(p.grad).all():bad.append(f"{twin}.{name}")
    result["non_finite_gradient_parameter_names"]=sorted(set(bad));return result

def optimizer_non_finite(optimizer):
    bad=[]
    for parameter,state in optimizer.state.items():
        for key,value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():bad.append({"parameter_id":id(parameter),"state":key})
    return bad

def numpy_batch_stats(batch):
    return {key:tensor_stats(batch[key]) for key in ("observations","previous_actions","progress","actions","returns")}

def dump_failure(output,step,stage,batch,details,model,optimizer):
    output=Path(output)/"diagnostics";output.mkdir(parents=True,exist_ok=True);stem=output/f"failure_step_{int(step):08d}"
    metadata={key:np.asarray(batch[key]).tolist() for key in ("policy","seed","episode_id","start","stop") if key in batch}
    report={"step":int(step),"first_non_finite_stage":stage,"batch_metadata":metadata,"input_statistics":numpy_batch_stats(batch),"details":details}
    stem.with_suffix(".json").write_text(json.dumps(report,indent=2,allow_nan=True)+"\n")
    np.savez_compressed(stem.with_suffix(".npz"),**{key:value for key,value in batch.items() if isinstance(value,np.ndarray)})
    torch.save({"step":int(step),"stage":stage,"critic_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict()},stem.with_suffix(".pth"))
    raise RuntimeError(f"Stage2.2 non-finite detected at step {step}, stage={stage}; diagnostic={stem}.json")

def ensure_finite(stage,named_values):
    bad=non_finite_names(named_values)
    if bad:raise FloatingPointError(f"{stage}: non-finite values in {bad}")

def finite_scalar(value):return math.isfinite(float(value))
