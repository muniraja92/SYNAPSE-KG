"""Colab driver: runs the training seeds in parallel lanes, prints progress, and after each seed
prints a compact checkpoint of its result (zlib+base64 pickle) into the cell output, so that a
finished seed survives a runtime disconnect (the notebook output is autosaved to Drive).
At the end it aggregates the seeds and prints the packed results (summary.json + figures).
Usage: python run_all.py "0,3" "1,4" "2"      (one argument per lane)
Restore after a disconnect: python run_all.py --restore ckpt.txt   (writes out/seed*.pkl from CKPT lines)"""
import sys, os, time, subprocess, pickle, zlib, base64, glob
import numpy as np

OUT = os.environ.get("OUT", "./out"); CH = 50000

def emit(s):
    r = pickle.load(open(f"{OUT}/seed{s}.pkl", 'rb'))
    if s != 0: r.pop('tsne_emb', None)          # aggregate.py uses the t-SNE rows of the first seed only
    b = base64.b64encode(zlib.compress(pickle.dumps(r, protocol=4), 9)).decode()
    parts = [b[i:i + CH] for i in range(0, len(b), CH)]
    for k, p in enumerate(parts):
        print(f"CKPT seed{s} {k + 1}/{len(parts)} {p}", flush=True)
    print(f"CKPT_END seed{s} chars={len(b)}", flush=True)

def restore(path):
    chunks = {}
    for line in open(path):
        if line.startswith('CKPT seed'):
            _, name, kn, data = line.split()[:4]; k, n = map(int, kn.split('/'))
            chunks.setdefault(name, {})[k] = (n, data)
    for name, d in chunks.items():
        n = next(iter(d.values()))[0]
        if len(d) != n: print('incomplete', name); continue
        r = pickle.loads(zlib.decompress(base64.b64decode(''.join(d[k][1] for k in range(1, n + 1)))))
        pickle.dump(r, open(f"{OUT}/{name}.pkl", 'wb')); print('restored', name)

if __name__ == '__main__':
    if sys.argv[1] == '--restore': restore(sys.argv[2]); sys.exit()
    lanes = sys.argv[1:]; os.makedirs('logs', exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    procs = []
    for j, lane in enumerate(lanes):
        procs.append(subprocess.Popen(f"python -u synapse_v3.py {' '.join(lane.split(','))} > logs/lane{j}.log 2>&1",
                                      shell=True, env=env))
    wanted = sorted(int(x) for lane in lanes for x in lane.split(','))
    done = {s for s in wanted if os.path.exists(f"{OUT}/seed{s}.pkl")}
    for s in sorted(done): emit(s)
    t0 = time.time(); last = 0
    while True:
        for s in wanted:
            if s not in done and os.path.exists(f"{OUT}/seed{s}.pkl"):
                done.add(s); emit(s)
        alive = [p.poll() is None for p in procs]
        if time.time() - last > 300 or not any(alive):
            last = time.time(); print(f"[{(time.time() - t0) / 60:.0f} min] done={sorted(done)}", flush=True)
            for j in range(len(lanes)):
                tail = open(f"logs/lane{j}.log").read().strip().splitlines()[-1:] if os.path.exists(f"logs/lane{j}.log") else []
                print(f"   lane{j} ({lanes[j]}): {tail[0][:150] if tail else ''}", flush=True)
        if not any(alive): break
        time.sleep(20)
    for j in range(len(lanes)):
        if procs[j].returncode != 0:
            print(f"lane{j} FAILED rc={procs[j].returncode}"); print(open(f"logs/lane{j}.log").read()[-3000:])
    if set(wanted) <= done:
        r = subprocess.run('python aggregate.py && python pack_results.py', shell=True, capture_output=True, text=True)
        print(r.stdout[-4000:], r.stderr[-2000:], flush=True)
        if os.path.exists('results_b64.txt'):
            b = open('results_b64.txt').read(); parts = [b[i:i + CH] for i in range(0, len(b), CH)]
            for k, p in enumerate(parts): print(f"RES {k + 1}/{len(parts)} {p}", flush=True)
            print("RES_END", len(b), flush=True)
