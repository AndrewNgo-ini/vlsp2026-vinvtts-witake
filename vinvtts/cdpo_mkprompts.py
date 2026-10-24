import json, random, re
from collections import Counter, defaultdict
random.seed(37)
D="/data/vinvtts/data/vinv-tts"; TAGS=["sniff","throatclearing","breathing","laughter"]
hold=set(l.strip() for l in open("/data/vinvtts/runs/evalR/holdout.txt"))
pub=set(json.loads(l)["text"] for f in ["A","B"] for l in open(f"/data/vinvtts/runs/public/{f}.jsonl"))
pubn=set(" ".join(re.sub(r"\[[a-z]+\]"," ",t).split()) for t in pub)
import soundfile as sf
rows=[dict(r,dur=sf.info(r["audio"]).duration) for r in map(json.loads,open(f"{D}/train.jsonl"))]
by=defaultdict(list)
for r in rows:
    if 3<=r["dur"]<=9 and r["audio"] not in hold: by[r["speaker"]].append(r)
spk=sorted(by)
strip=lambda t:" ".join(re.sub(r"\[[a-z]+\]"," ",t).split())
sents=[s.strip() for s in open("/data/vinvtts/data/crawl2/football_talk_sentences.txt") if 8<=len(s.split())<=28]
used=set(strip(json.loads(l)["text"]) for f in ["../prompts.jsonl","../prompts2.jsonl","../r2/prompts.jsonl","../r2/prompts_b.jsonl"] for l in open(f))
random.shuffle(sents); sents=[s for s in sents if s not in pubn and s not in used]; print("avail",len(sents))
def ins(ws,tag):
    u=random.random()
    if u<.15: return [f"[{tag}]"]+ws
    if u<.30: return ws+[f"[{tag}]"]
    i=random.randint(2,len(ws)-2); return ws[:i]+[f"[{tag}]"]+ws[i:]
out=[]; si=0
def mk(prefix,src,n,refsel):
    global si
    for k in range(n):
        s=sents[si]; si+=1; tag=TAGS[k%4]; ws=ins(s.split(),tag)
        if random.random()<.07:
            t2=random.choice(TAGS); i=random.randint(1,len(ws)-1)
            if not ws[i].startswith("[") and not ws[i-1].startswith("["): ws=ws[:i]+[f"[{t2}]"]+ws[i:]
        sp,a,rt=refsel()
        out.append({"id":f"{prefix}{len(out):04d}","text":" ".join(ws),"speaker":sp,"ref":a,"ref_text":rt,"src":src,"cls":tag})
def rnd():
    sp=random.choice(spk); x=random.choice(by[sp]); return sp,x["audio"],strip(x["text"])
# clean read-speech-like refs: top DNSMOS train clips (Track-B-like), different pool from r2 top-120
tc=[r for r in map(json.loads,open("/data/vinvtts/runs/cdpo/r2/trainclips.scored.jsonl")) if 3<=r["dur"]<=9 and r["wer"]<=.1 and r["gen"] not in hold]
tc.sort(key=lambda r:-r["pmos"]); clean=tc[:200]
def cln():
    x=random.choice(clean); return x["speaker"],x["gen"],strip(x["text"])
mk("gf","football",800,rnd)
mk("gb","clean",500,cln)
tt=defaultdict(list)
for r in rows:
    tg=set(re.findall(r"\[([a-z]+)\]",r["text"]))
    if len(tg)==1 and r["audio"] not in hold and 5<=len(r["text"].split())<=40 and r["speaker"] in by and strip(r["text"]) not in pubn: tt[tg.pop()].append(r)
for tag in TAGS:
    random.shuffle(tt[tag])
    for r in tt[tag][200:260]:
        c=[x for x in by[r["speaker"]] if x["audio"]!=r["audio"]]
        if not c: continue
        x=random.choice(c)
        out.append({"id":f"gt{len(out):04d}","text":r["text"],"speaker":r["speaker"],"ref":x["audio"],"ref_text":strip(x["text"]),"src":"train","cls":tag})
assert not any(o["text"] in pub for o in out)
random.shuffle(out)
open("prompts.jsonl","w").writelines(json.dumps(x,ensure_ascii=False)+"\n" for x in out)
print(len(out),Counter(x["src"] for x in out),Counter(x["cls"] for x in out),clean[-1]["pmos"])
