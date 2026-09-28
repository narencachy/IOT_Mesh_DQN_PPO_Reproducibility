import math, time, json, random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import wilcoxon
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

OUT=Path('/mnt/data/iot_mesh_rl_results'); OUT.mkdir(exist_ok=True)
GRAPH_CACHE={}
DEVICE='cpu'
torch.set_num_threads(1)

ACTIONS=4
STATE_DIM=12
DT=0.5
THRESH=0.60

@dataclass
class StepStats:
    reward: float
    new_inf: int
    blocked: int
    eligible: int
    benign_tp: float
    unnecessary: int
    candidate_infected: bool
    action: int

class MeshContainmentEnv:
    def __init__(self, n=50, seed=0, max_steps=40):
        self.n=n; self.seed=seed; self.max_steps=max_steps
        self.rng=np.random.default_rng(seed)
        self.G=None
        self.reset()
    def _graph(self):
        # generate connected random geometric-like graph from k-NN over random points
        pts=self.rng.random((self.n,2))
        G=nx.Graph(); G.add_nodes_from(range(self.n))
        # connect 5 nearest neighbors symmetrically; enough for connectivity usually
        D=((pts[:,None,:]-pts[None,:,:])**2).sum(-1)
        for i in range(self.n):
            nnidx=np.argsort(D[i])[1:6]
            for j in nnidx: G.add_edge(i,int(j))
        # connect components if needed by closest pair
        while not nx.is_connected(G):
            comps=[list(c) for c in nx.connected_components(G)]
            c0=comps[0]; c1=comps[1]
            best=min(((D[i,j],i,j) for i in c0 for j in c1), key=lambda x:x[0])
            G.add_edge(best[1],best[2])
        self.pos={i:pts[i] for i in range(self.n)}
        return G
    def reset(self):
        self.rng=np.random.default_rng(self.seed + getattr(self,'episode',0)*9973)
        self.episode=getattr(self,'episode',0)+1
        if self.G is None:
            key=(self.n,self.seed)
            if key in GRAPH_CACHE:
                self.G,self.pos,self.bet,self.deg,self.frag=GRAPH_CACHE[key]
            else:
                self.G=self._graph()
                self.bet=np.array(list(nx.betweenness_centrality(self.G, normalized=True).values()),dtype=np.float32)
                self.deg=np.array([self.G.degree(i)/(self.n-1) for i in range(self.n)],dtype=np.float32)
                self.frag=np.zeros(self.n,dtype=np.float32)
                arts=set(nx.articulation_points(self.G))
                for c in arts:
                    self.frag[c]=0.25
                GRAPH_CACHE[key]=(self.G,self.pos,self.bet,self.deg,self.frag)
        self.load=self.rng.uniform(0.3,1.0,self.n).astype(np.float32)
        self.vul=self.rng.random(self.n)<0.65
        self.infected=np.zeros(self.n,dtype=bool)
        vul_idx=np.where(self.vul)[0]
        init=self.rng.choice(vul_idx,size=min(2,len(vul_idx)),replace=False)
        self.infected[init]=True
        self.action_state=np.zeros(self.n,dtype=np.int8)
        self.action_ttl=np.zeros(self.n,dtype=np.int8)
        self.prev_action=0
        self.t=0
        self.total_blocked=0; self.total_eligible=0; self.unnecessary_nodes=set(); self.tp_history=[]
        self.zero_new_streak=0; self.contained_at=None
        return self._state()
    def _scores(self):
        # controlled detector: same across policies for same env seed/episode RNG state
        # infected higher, benign lower; sd provides realistic FPR/FNR.
        mu=np.where(self.infected,0.78,0.22)
        scores=np.clip(mu + self.rng.normal(0,0.18,self.n),0,1)
        return scores
    def _candidate(self,scores):
        flagged=np.where(scores>=THRESH)[0]
        if len(flagged)==0:
            return int(np.argmax(scores))
        # sample flagged alerts proportionally to score so detector false positives enter the response stream
        w=scores[flagged]**3; w=w/w.sum(); return int(self.rng.choice(flagged,p=w))
    def _state(self):
        # state generation advances detector noise; acceptable within each episode only once per decision
        scores=self._scores(); c=self._candidate(scores); self.current_scores=scores; self.current_c=c
        neigh=list(self.G.neighbors(c))
        inf_neigh=np.mean(self.infected[neigh]) if neigh else 0.0
        global_inf=self.infected.mean()
        # connectivity margin if candidate were removed (precomputed for the fixed training graph)
        gcc=1.0-float(self.frag[c])
        last=self.prev_action/(ACTIONS-1)
        s=np.array([
            scores[c], self.bet[c], self.deg[c], self.load[c], inf_neigh, global_inf,
            gcc, float(self.infected[c]), float(self.vul[c]), self.t/self.max_steps,
            last, np.mean(self.load)
        ], dtype=np.float32)
        return s
    @staticmethod
    def action_effect(a):
        # malicious propagation multiplier, benign retention, control severity
        if a==0: return 1.0,1.0,0.0
        if a==1: return 0.45,0.92,0.25
        if a==2: return 0.08,0.72,0.65
        return 0.0,0.0,1.0
    def step(self,a):
        c=self.current_c; scores=self.current_scores
        # decay previously installed controls, then install/refresh the chosen action for three decision intervals
        self.action_ttl=np.maximum(self.action_ttl-1,0); self.action_state[self.action_ttl==0]=0
        self.action_state[c]=a; self.action_ttl[c]=3 if a>0 else 0
        prop_mult=np.ones(self.n,dtype=float); benign_ret=np.ones(self.n,dtype=float)
        for j in range(self.n):
            mm,bb,_=self.action_effect(int(self.action_state[j])); prop_mult[j]=mm; benign_ret[j]=bb
        m,bret,sev=self.action_effect(a)
        if (not self.infected[c]) and a>=2:
            self.unnecessary_nodes.add(c)
        # propagation attempts from infected nodes to susceptible vulnerable neighbors
        new=[]; blocked=0; elig=0
        for j in np.where(self.infected)[0]:
            for i in self.G.neighbors(int(j)):
                if self.infected[i] or (not self.vul[i]): continue
                elig+=1
                base=0.18*(0.65+0.35*self.load[j])
                eff=base*prop_mult[j]
                # if source is candidate and action suppressed, blocked amount counts probabilistically via shared draw
                u=self.rng.random()
                if u < base:
                    if u < eff:
                        new.append(i)
                    else:
                        blocked += 1
        if new:
            self.infected[np.unique(new)]=True
        new_inf=len(np.unique(new))
        self.total_blocked += blocked; self.total_eligible += elig
        if new_inf==0: self.zero_new_streak+=1
        else: self.zero_new_streak=0
        if self.contained_at is None and self.zero_new_streak>=3:
            self.contained_at=(self.t-1)*DT if self.t>=2 else self.t*DT
        # benign throughput weighted by node load; infected traffic excluded from benign denominator
        benign=~self.infected
        denom=float(np.sum(self.load[benign]))+1e-9
        btp=float(np.sum(self.load[benign]*benign_ret[benign])/denom) if benign.any() else 0.0
        # partition penalty if hard-dropping high-centrality node could disconnect graph
        partition=float(self.frag[c]) if a==3 else 0.0
        # reward: suppress spread, preserve benign throughput, avoid damaging central nodes and action churn
        malicious_gain = blocked - 1.8*new_inf - 0.015*self.infected.sum()
        qos_pen = 3.5*(1-btp) + 2.5*partition + (1.2*self.bet[c]*sev if not self.infected[c] else 0)
        reconfig = 0.04 if a!=self.prev_action else 0.0
        reward=2.2*malicious_gain + 0.8*btp - 2.4*qos_pen - reconfig
        self.prev_action=a; self.t+=1
        done=self.t>=self.max_steps or self.infected.mean()>0.65
        state=self._state() if not done else np.zeros(STATE_DIM,dtype=np.float32)
        return state,float(reward),done,StepStats(reward,new_inf,blocked,elig,btp,len(self.unnecessary_nodes),bool(self.infected[c]),a)
    def metrics(self):
        # CSR: blocked attack opportunities / realized eligible attack opportunities approximated by blocked/(blocked+new infection events)
        # We record total eligible, but policy relevant CSR better as blocked / (blocked + infections)
        infected_final=int(self.infected.sum())
        csr=self.total_blocked/max(1,self.total_blocked+max(0,infected_final-2))
        latency=self.contained_at if self.contained_at is not None else self.max_steps*DT
        unnecessary=100*len(self.unnecessary_nodes)/self.n
        # throughput history not stored; approximate from final action not enough, so caller aggregates step btp
        return latency,100*csr,unnecessary,infected_final

class QNet(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(STATE_DIM,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,ACTIONS))
    def forward(self,x): return self.net(x)
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__(); self.body=nn.Sequential(nn.Linear(STATE_DIM,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh()); self.pi=nn.Linear(128,ACTIONS); self.v=nn.Linear(128,1)
    def forward(self,x):
        h=self.body(x); return self.pi(h),self.v(h).squeeze(-1)

class Replay:
    def __init__(self,cap=50000): self.cap=cap; self.data=[]; self.p=0
    def add(self,*x):
        if len(self.data)<self.cap: self.data.append(x)
        else: self.data[self.p]=x; self.p=(self.p+1)%self.cap
    def sample(self,n,rng):
        idx=rng.choice(len(self.data),size=n,replace=False); return [self.data[i] for i in idx]

def train_dqn(seed,episodes=700):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env=MeshContainmentEnv(seed=seed*100); net=QNet(); target=QNet(); target.load_state_dict(net.state_dict())
    opt=optim.Adam(net.parameters(),lr=7e-4); buf=Replay(30000); rng=np.random.default_rng(seed)
    rewards=[]; gamma=.97; batch=64
    for ep in range(episodes):
        s=env.reset(); total=0; eps=max(.05,1.0-0.95*ep/(episodes*.7))
        done=False
        while not done:
            if rng.random()<eps: a=int(rng.integers(ACTIONS))
            else:
                with torch.no_grad(): a=int(net(torch.tensor(s).unsqueeze(0)).argmax(1).item())
            ns,r,done,st=env.step(a); buf.add(s,a,r,ns,float(done)); s=ns; total+=r
            if len(buf.data)>=batch:
                samp=buf.sample(batch,rng)
                S=torch.tensor(np.stack([x[0] for x in samp]),dtype=torch.float32); A=torch.tensor([x[1] for x in samp]); R=torch.tensor([x[2] for x in samp],dtype=torch.float32); NS=torch.tensor(np.stack([x[3] for x in samp]),dtype=torch.float32); D=torch.tensor([x[4] for x in samp],dtype=torch.float32)
                q=net(S).gather(1,A[:,None]).squeeze(1)
                with torch.no_grad():
                    na=net(NS).argmax(1); nq=target(NS).gather(1,na[:,None]).squeeze(1); y=R+gamma*(1-D)*nq
                loss=nn.functional.smooth_l1_loss(q,y); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),5); opt.step()
            # soft target
            with torch.no_grad():
                for p,tp in zip(net.parameters(),target.parameters()): tp.data.mul_(.995).add_(.005*p.data)
        rewards.append(total)
    return net,np.array(rewards)

def train_ppo(seed,episodes=700):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env=MeshContainmentEnv(seed=seed*100); ac=ActorCritic(); opt=optim.Adam(ac.parameters(),lr=3e-4); gamma=.97; lam=.95; clip=.2
    rewards=[]
    for ep in range(episodes):
        s=env.reset(); traj=[]; total=0; done=False
        while not done:
            st=torch.tensor(s,dtype=torch.float32).unsqueeze(0); logits,v=ac(st); dist=Categorical(logits=logits); a=dist.sample(); logp=dist.log_prob(a)
            ns,r,done,ss=env.step(int(a.item())); traj.append((s,int(a.item()),r,ns,float(done),float(logp.item()),float(v.item()))); total+=r; s=ns
        # compute GAE
        vals=np.array([x[6] for x in traj]+[0.0]); rs=np.array([x[2] for x in traj]); ds=np.array([x[4] for x in traj]); adv=np.zeros(len(traj)); gae=0
        for t in reversed(range(len(traj))):
            delta=rs[t]+gamma*vals[t+1]*(1-ds[t])-vals[t]; gae=delta+gamma*lam*(1-ds[t])*gae; adv[t]=gae
        ret=adv+vals[:-1]
        S=torch.tensor(np.stack([x[0] for x in traj]),dtype=torch.float32); A=torch.tensor([x[1] for x in traj]); OLD=torch.tensor([x[5] for x in traj],dtype=torch.float32); ADV=torch.tensor(adv,dtype=torch.float32); RET=torch.tensor(ret,dtype=torch.float32)
        ADV=(ADV-ADV.mean())/(ADV.std()+1e-8)
        for _ in range(4):
            logits,v=ac(S); dist=Categorical(logits=logits); lp=dist.log_prob(A); ratio=(lp-OLD).exp(); s1=ratio*ADV; s2=torch.clamp(ratio,1-clip,1+clip)*ADV
            loss=-torch.min(s1,s2).mean()+.5*((v-RET)**2).mean()-.01*dist.entropy().mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ac.parameters(),5); opt.step()
        rewards.append(total)
    return ac,np.array(rewards)

def act_dqn(model,s):
    with torch.no_grad(): return int(model(torch.tensor(s,dtype=torch.float32).unsqueeze(0)).argmax(1).item())
def act_ppo(model,s):
    with torch.no_grad(): return int(torch.argmax(model(torch.tensor(s,dtype=torch.float32).unsqueeze(0))[0],dim=1).item())
def act_static(env,s): return 3 if s[0]>=0.75 else 0
def act_rate(env,s): return 1 if s[0]>=0.75 else 0

def evaluate(policy, model, seeds):
    rows=[]
    for seed in seeds:
        env=MeshContainmentEnv(seed=seed); s=env.reset(); done=False; bt=[]
        while not done:
            if policy=='dqn': a=act_dqn(model,s)
            elif policy=='ppo': a=act_ppo(model,s)
            elif policy=='static': a=act_static(env,s)
            else: a=act_rate(env,s)
            s,r,done,st=env.step(a); bt.append(st.benign_tp)
        lat,csr,unnec,finalinf=env.metrics(); rows.append(dict(seed=seed,method=policy,latency=lat,csr=csr,unnecessary=unnec,benign_tp=100*np.mean(bt),final_infected=finalinf))
    return rows

def ci95(x):
    x=np.asarray(x,float); return 1.96*x.std(ddof=1)/math.sqrt(len(x))

def main():
    train_seeds=[11,22,33,44,55]
    episodes=120
    t0=time.time()
    models_dqn={}; models_ppo={}; curves_dqn=[]; curves_ppo=[]
    for seed in train_seeds:
        d,dc=train_dqn(seed,episodes=episodes); p,pc=train_ppo(seed,episodes=episodes)
        models_dqn[seed]=d; models_ppo[seed]=p; curves_dqn.append(dc); curves_ppo.append(pc)
        torch.save(d.state_dict(),OUT/f'dqn_{seed}.pt'); torch.save(p.state_dict(),OUT/f'ppo_{seed}.pt')

    # 50 paired evaluations: 10 disjoint scenario seeds per independently trained checkpoint.
    rows=[]
    for j,seed in enumerate(train_seeds):
        scenarios=list(range(9001+j*10,9011+j*10))
        for method,model in [('static',None),('rate',None),('dqn',models_dqn[seed]),('ppo',models_ppo[seed])]:
            rr=evaluate(method,model,scenarios)
            for r in rr:
                r['train_seed']=seed; r['pair_id']=f'{seed}-{r["seed"]}'
            rows += rr
    df=pd.DataFrame(rows); df.to_csv(OUT/'evaluation_runs.csv',index=False)

    summary=[]
    for method,g in df.groupby('method'):
        row={'method':method,'n':len(g)}
        for col in ['latency','csr','unnecessary','benign_tp','final_infected']:
            row[col+'_mean']=g[col].mean(); row[col+'_sd']=g[col].std(ddof=1); row[col+'_ci95']=ci95(g[col])
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)

    piv=df.pivot(index='pair_id',columns='method',values=['latency','csr','unnecessary','benign_tp'])
    tests=[]
    for metric in ['latency','csr','unnecessary','benign_tp']:
        x=piv[(metric,'ppo')].values; y=piv[(metric,'dqn')].values
        stat,p=wilcoxon(x,y,alternative='two-sided',zero_method='wilcox')
        diff=x-y; nz=diff[diff!=0]
        ranks=pd.Series(np.abs(nz)).rank(method='average').values if len(nz) else np.array([])
        wp=ranks[nz>0].sum() if len(nz) else 0; wm=ranks[nz<0].sum() if len(nz) else 0
        rrb=(wp-wm)/(wp+wm) if wp+wm else 0
        tests.append(dict(metric=metric,dqn=float(y.mean()),ppo=float(x.mean()),delta=float(diff.mean()),p=float(p),r_rank_biserial=float(rrb)))
    ps=np.array([t['p'] for t in tests]); order=np.argsort(ps); adj=np.zeros(len(ps)); running=0
    for rank,idx in enumerate(order):
        val=(len(ps)-rank)*ps[idx]; running=max(running,val); adj[idx]=min(1,running)
    for t,a in zip(tests,adj): t['p_holm']=float(a)
    pd.DataFrame(tests).to_csv(OUT/'paired_tests.csv',index=False)

    # Convergence figure from all five training seeds.
    D=np.stack(curves_dqn); P=np.stack(curves_ppo); x=np.arange(1,episodes+1)
    def smooth(a,w=15): return np.vstack([pd.Series(r).rolling(w,min_periods=1).mean().values for r in a])
    Ds,Ps=smooth(D),smooth(P)
    fig,ax=plt.subplots(figsize=(6.0,3.2))
    for arr,label in [(Ds,'DQN'),(Ps,'PPO')]:
        mean=arr.mean(0); ci=1.96*arr.std(0,ddof=1)/math.sqrt(arr.shape[0]); ax.plot(x,mean,label=label); ax.fill_between(x,mean-ci,mean+ci,alpha=.18)
    ax.set_xlabel('Training episode'); ax.set_ylabel('Episode reward'); ax.legend(frameon=False); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT/'convergence.png',dpi=220); plt.close(fig)
    pd.DataFrame({'episode':x,'dqn_mean':Ds.mean(0),'dqn_ci95':1.96*Ds.std(0,ddof=1)/math.sqrt(5),'ppo_mean':Ps.mean(0),'ppo_ci95':1.96*Ps.std(0,ddof=1)/math.sqrt(5)}).to_csv(OUT/'convergence.csv',index=False)

    # Fixed detector operating point used by every policy.
    rng=np.random.default_rng(12345); n=500000; y=rng.random(n)<.2
    scores=np.clip(np.where(y,.78,.22)+rng.normal(0,.18,n),0,1); pred=scores>=THRESH
    tpr=(pred & y).sum()/y.sum(); fpr=(pred & ~y).sum()/(~y).sum(); fnr=((~pred)&y).sum()/y.sum(); precision=(pred&y).sum()/pred.sum()
    json.dump({'threshold':THRESH,'tpr':float(tpr),'fpr':float(fpr),'fnr':float(fnr),'precision':float(precision),'prevalence':0.2,'n_mc':n},open(OUT/'detector.json','w'),indent=2)

    # Controller-side state construction + inference only; no OpenFlow switch-installation timing.
    dqn=models_dqn[22]; ppo=models_ppo[55]; overhead=[]
    for nnode in [25,50,75,100]:
        env=MeshContainmentEnv(n=nnode,seed=70000+nnode,max_steps=10); env.reset()
        for name,model,act in [('DQN',dqn,act_dqn),('PPO',ppo,act_ppo)]:
            vals=[]
            for _ in range(1000):
                t=time.perf_counter_ns(); st=env._state(); _=act(model,st); vals.append((time.perf_counter_ns()-t)/1e6)
            overhead.append(dict(nodes=nnode,agent=name,mean_ms=float(np.mean(vals)),p95_ms=float(np.percentile(vals,95)),p99_ms=float(np.percentile(vals,99))))
    pd.DataFrame(overhead).to_csv(OUT/'controller_overhead.csv',index=False)
    info={}
    for name,model in [('DQN',dqn),('PPO',ppo)]:
        params=sum(p.numel() for p in model.parameters()); bytes_=sum(p.numel()*p.element_size() for p in model.parameters()); info[name]={'parameters':params,'weights_kb':bytes_/1024}
    info['training_seconds']=time.time()-t0
    json.dump(info,open(OUT/'model_info.json','w'),indent=2)
    print(pd.read_csv(OUT/'summary.csv').to_string(index=False)); print(pd.read_csv(OUT/'paired_tests.csv').to_string(index=False))

if __name__=='__main__': main()
