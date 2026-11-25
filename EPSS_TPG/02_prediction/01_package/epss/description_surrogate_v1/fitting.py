"""Day60 optimization/selection only; best-only deterministic checkpoint replay."""
import random
import numpy as np
import torch
from tqdm import tqdm

from ..training_only_pooled_v1.fitting import seed_all, rmse
from ..reddit_development_v1.common import save_torch
from . import data, model


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.writing');temp.write_text(data.a.dump(value));temp.replace(path)


def reload(root,identity,p,x,y):
    cp=torch.load(root/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    hist=data.a.read(root/'history.json')
    data.a.require(cp['identity']==identity and hist,'Checkpoint identity/history differs')
    best=min(hist,key=lambda r:r['selection_rmse'])
    data.a.require(cp['epoch']==best['epoch'] and cp['history']==hist[:cp['epoch']], 'Not earliest strict best')
    seed_all(identity['seed']);net=model.DescriptionRegressor(p['hidden']);net.load_state_dict(cp['model'])
    data.a.require(rmse(y,model.predict(net,x))==cp['selection_rmse'],'Selection replay differs')
    return net,cp


def fit(root,identity,p,x,y,sx,sy,resume=False):
    if root.exists():
        data.a.require(resume and data.a.read(root/'identity.json')==identity,'Partial fit needs matching --resume')
    else:write(root/'identity.json',identity)
    seed=identity['seed'];fold=identity['fold'];seed_all(seed)
    net=model.DescriptionRegressor(p['hidden'])
    opt=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    checkpoint=root/'best_checkpoint.pt';hp=root/'history.json'
    expected=data.a.read(hp) if hp.exists() else []
    hist=[];start=1;best=float('inf');stale=0
    if checkpoint.exists():
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        data.a.require(cp['identity']==identity,'Resume checkpoint identity changed')
        net.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer'])
        torch.set_rng_state(cp['torch_rng']);np.random.set_state(cp['numpy_rng']);random.setstate(cp['python_rng'])
        hist=cp['history'];start=cp['epoch']+1;best=cp['selection_rmse']
        n=min(len(hist),len(expected));data.a.require(hist[:n]==expected[:n],'Checkpoint/history prefix mismatch')
        if len(expected)<len(hist):expected=hist.copy();write(hp,hist)
    yy=torch.as_tensor(y,dtype=torch.float32)
    for epoch in tqdm(range(start,p['epochs']+1),desc=f'DS01 fold{fold} seed{seed}',unit='epoch',leave=False):
        net.train();order=np.random.default_rng(seed*1000003+fold*1009+epoch).permutation(len(y));total=0.
        for i in tqdm(range(0,len(y),p['batch_size']),desc='Optimize description batches',unit='batch',leave=False):
            take=order[i:i+p['batch_size']];opt.zero_grad(set_to_none=True)
            pred=net({'text':x[take]});loss=((pred-yy[take])**2).mean()
            data.a.require(bool(torch.isfinite(loss)),'Nonfinite loss')
            loss.backward();opt.step();total+=float(loss.detach())*len(take)
        score=rmse(sy,model.predict(net,sx));improved=score<best
        if improved:best,stale=score,0
        else:stale+=1
        record={'epoch':epoch,'optimization_mse':total/len(y),'selection_rmse':score,
                'best_selection_rmse':best,'patience':stale}
        if epoch<=len(expected):data.a.require(record==expected[epoch-1],'Interrupted epoch did not replay')
        hist.append(record)
        if improved:
            save_torch(checkpoint,{'identity':identity,'model':net.state_dict(),'optimizer':opt.state_dict(),
                       'epoch':epoch,'selection_rmse':score,'history':hist.copy(),
                       'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()})
        write(hp,hist)
        tqdm.write(f'[DS01] fold={fold} seed={seed} epoch={epoch} MSE={total/len(y):.8f} '
                   f'selection_RMSE={score:.8f} best={best:.8f} patience={stale}/{p["patience"]}')
        if stale>=p['patience']:break
    return reload(root,identity,p,sx,sy)
