"""Round-3 multi-objective pairs. Tag judges (v3 tag-ASR and audit judge NOT used):
 laughter: nvdet, LS, PSED-laughter ; breathing: nvdet, PSED-breathing ; sniff/throat: nvdet, PSED rare clf.
 realised = >=2 judge votes (all votes when only 2 judges); notreal = <=1 vote (and for 2-judge classes: 0 votes).
 TAG pairs : chosen all tags realised, WER<=.1; rejected: target tag not realised; AND utmos_w > utmos_l (or dnsmos_w > dnsmos_l)
             AND ss_w >= ss_l.
 SS pairs  : clean read-speech (gb*) prompts; both all-realised, WER<=.1; ss_w - ss_l >= .04, utmos_w >= utmos_l.
 Balance   : tag pairs capped per class at 1.2x rarest; SS pairs ~ 1/4 of total."""
import json, random, glob
from collections import defaultdict, Counter
random.seed(3); C="/data/vinvtts/runs/cdpo/r3/"; PT=.5
cand={}; tj=defaultdict(dict)
def J(f): return map(json.loads,open(f))
for b in sorted(glob.glob(C+"q?_gs?.scored.jsonl")):
    b=b[:-13]
    for r in J(b+".scored.jsonl"): cand[r["id"]+b[-3:]]=dict(r,gs=b[-3:])
    ut={r["id"]:r["utmos"] for r in J(b+".utmos.jsonl")}
    for r in J(b+".scored.jsonl"): cand[r["id"]+b[-3:]]["utmos"]=ut[r["id"]]
    k=lambda r:r["id"]+b[-3:]
    for r in J(b+".nvdet.jsonl"): tj[k(r)].setdefault(r["gap"],{"tag":r["tag"]}).update(nd=r["p_tag"],pred=r["pred"])
    for r in J(b+".laugh.jsonl"): tj[k(r)].setdefault(r["gap"],{}).update(ls=r["p"])
    for r in J(b+".rare.jsonl"): tj[k(r)].setdefault(r["gap"],{}).update(ph=r["hit"])
    for r in J(b+"_bl.psed.jsonl"):
        ci=2 if r["tag"]=="laughter" else 3; tj[k(r)].setdefault(r["gap"],{}).update(pb=sum(m[ci] for m in r["probs"])/3)
def votes(j):
    t=j.get("tag"); v=[j.get("pred")==t and j.get("nd",0)>=.5]
    if t=="laughter": v+=[j.get("ls",0)>=.5, j.get("pb",0)>=PT]
    elif t=="breathing": v+=[j.get("pb",0)>=PT]
    else: v+=[bool(j.get("ph"))]
    return sum(v), len(v)
by=defaultdict(list)
for i,r in cand.items():
    r["st"]=[(j["tag"],)+votes(j) for j in tj[i].values() if "tag" in j]
    r["ok"]=[n>=2 and (n==m or m==3) for _,n,m in r["st"]]
    by[r["base_id"]].append(r)
real=lambda c:c["st"] and all(c["ok"])
def notreal(c,cls): return any(t==cls and not o for (t,n,m),o in zip(c["st"],c["ok"]))
tagp=defaultdict(list); ssp=[]
for b,cs in by.items():
    good=[c for c in cs if real(c) and c["wer"]<=.1]
    if not good: continue
    w=max(good,key=lambda c:c["utmos"]+.5*c["pmos"]/5*5/5+2*c["ss"]-2*c["wer"])
    good.sort(key=lambda c:-(c["utmos"]+2*c["ss"]-2*c["wer"])); w=good[0]; np_=0; used=set()
    for cls in set(t for t,_,_ in w["st"]):
        for wc in good:
            bad=[c for c in cs if c["gen"] not in used and notreal(c,cls) and (c["utmos"]<wc["utmos"] or c["pmos"]<wc["pmos"]) and c["ss"]<=wc["ss"]]
            if bad:
                l=random.choice(bad); used.add(l["gen"]); tagp[cls].append(dict(kind="tag",cls=cls,w=wc,l=l)); np_+=1
                if np_>=2: break
        if np_>=2: break
    if np_==0 and b.startswith("gb") and len(good)>=2:
        g2=[c for c in good if w["ss"]-c["ss"]>=.04 and c["utmos"]<=w["utmos"]]
        if g2: ssp.append(dict(kind="ss",cls=w["cls"],w=w,l=min(g2,key=lambda c:c["ss"])))
cnt={k:len(v) for k,v in tagp.items()}; print("raw tag",cnt,"raw ss",len(ssp))
cap=int(min(cnt.values())*1.2); out=[]
for k,v in tagp.items(): random.shuffle(v); out+=v[:cap]
random.shuffle(ssp); out+=ssp[:max(len(out)//3,1)]
random.shuffle(out)
rows=[{"id":p["w"]["base_id"],"kind":p["kind"],"cls":p["cls"],"text":p["w"]["text"],"ref":p["w"]["ref"],"ref_text":p["w"]["ref_text"],"chosen":p["w"]["gen"],"rejected":p["l"]["gen"],
       "d_utmos":p["w"]["utmos"]-p["l"]["utmos"],"d_pmos":p["w"]["pmos"]-p["l"]["pmos"],"d_ss":p["w"]["ss"]-p["l"]["ss"],"d_wer":p["w"]["wer"]-p["l"]["wer"]} for p in out]
open(C+"pairs.jsonl","w").writelines(json.dumps(p,ensure_ascii=False)+"\n" for p in rows)
for kd in ["tag","ss"]:
    R=[r for r in rows if r["kind"]==kd]
    if R: print(kd,len(R),Counter(r["cls"] for r in R),{x:round(sum(r[x] for r in R)/len(R),3) for x in ["d_utmos","d_pmos","d_ss","d_wer"]})
d=defaultdict(lambda:[0,0])
for r in cand.values():
    for (t,n,m),o in zip(r["st"],r["ok"]): d[(r["gs"],t)][0]+=o; d[(r["gs"],t)][1]+=1
print("realised",{k:round(a/b,2) for k,(a,b) in sorted(d.items())}); print("total",len(rows))
