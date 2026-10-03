"""Acceptance of a kernel library (TZ_v100_kernel.md §7, tests 1 and 2) against the CPU reference.

    python3 acceptance/accept.py --so /opt/own/kernel.so [--tile 8x16|16x16|v100] [--tiles 1000000] [--out accept.json]
                                 [--dump-dir dir]

--tile: the hash tile the library is built for (miner.jobs.TILES): 8x16 (s0 §6), contiguous 16x16 (SOAT, which
refuses any other pattern with code 3) or v100 (kernels/v100, m8n8k4); both the reference and the miner of test 2
use it.

Test 1 -- exactness. The contract gives no transcripts out, so each case is run at the saturated bound
(2^256 - 1: every tile is a candidate) with a cap above the tiles of the call: the library returns the jackpot of
every tile, and a jackpot is keyed BLAKE3 of the whole transcript -- equal jackpots are equal transcripts (on a
mismatch the dump holds the reference transcripts). Candidates are compared as a set: the contract does not fix their
order. A second call of the same case at a middle bound (half the tiles) with cap 4 checks the comparison with the
bound (inclusive), *count and what is written over cap: count = reference count, min(count, 4) written, each one a
reference candidate. Cases: the edges +-127 (all +127, all -127, A' +127 x B' -127, a +-127 checkerboard, random
signs) at k 2048 / 4096 / 8192 first, then random cases until --tiles tiles: k of the three, m, n in 64..512,
a random row-tile range [lo, hi); operands either uniform in [-127, 127] or the miner's own (A = B = 0 plus noise,
seeds from a random header). The first mismatch writes the inputs and both outputs to --dump-dir and stops: FAIL.

Test 2 -- a share. Our host part (miner.jobs.Miner) with this library on an easy target (nbits 0x1e00ffff, ~1/256
tiles) until --shares shares: own verify (ref) -> the submitted base64 -> pearl_mining.verify_plain_proof_for_cert_version
(3, ...) if pearl_mining is importable, otherwise ref.verify again and "official": false.

The verdict (PASS / FAIL) and the details go to --out as JSON and the last stdout line; exit 0 on PASS, 1 on FAIL.
"""
import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

import miner  # noqa: E402,F401  (puts ref/ on sys.path)
import pearl_ref as R  # noqa: E402
from miner.jobs import TILES, Miner, Stats, mining_config  # noqa: E402
from miner.kernel import CpuRefBackend, KernelError, Operands, SoBackend  # noqa: E402
from miner.pool import Job  # noqa: E402

KS = (2048, 4096, 8192)
DIMS = (64, 128, 256, 512)
EDGES = ("all+127", "all-127", "a+127_b-127", "checker", "signs")
EASY = 0x1E00FFFF
SMALL_CAP = 4
CAP = 1 << 16          # > row tiles x col tiles of the largest case (512 x 512: 64 x 32)
ROW_TILE_STEP = {"v100": 4}   # kernels/v100 takes row tiles [lo, hi) with lo, hi multiples of 4 (pearl_api.h)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def header(rng):
    return R.header_bytes(0x20000000, rng.bytes(32), rng.bytes(32), int(time.time()), 0x1D00FFFF)


def edge_operands(kind, m, n, k, rng):
    if kind == "all+127":
        return np.full((m, k), 127, np.int8), np.full((n, k), 127, np.int8)
    if kind == "all-127":
        return np.full((m, k), -127, np.int8), np.full((n, k), -127, np.int8)
    if kind == "a+127_b-127":
        return np.full((m, k), 127, np.int8), np.full((n, k), -127, np.int8)
    if kind == "checker":
        s = lambda r: np.where((np.arange(r)[:, None] + np.arange(k)[None]) % 2, 127, -127).astype(np.int8)
        return s(m), s(n)
    if kind == "signs":
        return ((rng.integers(0, 2, (m, k)) * 254 - 127).astype(np.int8),
                (rng.integers(0, 2, (n, k)) * 254 - 127).astype(np.int8))
    raise ValueError(kind)


def make_case(i, kind, m, n, k, rng, tile="8x16"):
    """-> (description, Operands at the saturated bound)."""
    cfg = mining_config(k, tile=tile)
    if kind == "noise":
        backend = CpuRefBackend()
        mn = Miner(backend, None, Stats(backend.devices()), k=k, m=m, n=n, portion_rows=64, tile=tile)
        job = Job(f"accept{i}", header(rng), R.nbits_to_target(EASY), 1, 3, 1, 0.0)
        pw = mn.prepare_pass(mn.prepare_job(job), int(rng.integers(0, 1 << 20)), 1)
        a, bt, seed = pw.ops.a_noised, pw.ops.bt_noised, pw.ops.seed_a
    elif kind == "uniform":
        a = rng.integers(-127, 128, (m, k), dtype=np.int8)
        bt = rng.integers(-127, 128, (n, k), dtype=np.int8)
        seed = rng.bytes(32)
    else:
        a, bt = edge_operands(kind, m, n, k, rng)
        seed = rng.bytes(32)
    ops = Operands(cfg, np.ascontiguousarray(a), np.ascontiguousarray(bt), seed, R.U256_MAX, i, i,
                   cfg.rows.partition(m), cfg.cols.partition(n))
    return {"case": i, "kind": kind, "m": m, "n": n, "k": k, "tile": tile}, ops


def cases(tiles, rng, tile="8x16"):
    """Edges at every k first (whole range), then random cases until `tiles` tiles in all."""
    i, done = 0, 0
    for k in KS:
        for kind in EDGES:
            desc, ops = make_case(i, kind, 128, 128, k, rng, tile)
            yield desc, ops, 0, ops.row_tiles
            i += 1
            done += ops.row_tiles * len(ops.col_part)
    while done < tiles:
        k, m, n = int(rng.choice(KS)), int(rng.choice(DIMS)), int(rng.choice(DIMS))
        desc, ops = make_case(i, str(rng.choice(["uniform", "noise"])), m, n, k, rng, tile)
        step = ROW_TILE_STEP.get(tile, 1)
        lo = step * int(rng.integers(0, ops.row_tiles // step))
        hi = step * int(rng.integers(lo // step + 1, ops.row_tiles // step + 1))
        yield desc, ops, lo, hi
        i += 1
        done += (hi - lo) * len(ops.col_part)


def with_bound(ops, bound):
    return Operands(ops.cfg, ops.a_noised, ops.bt_noised, ops.seed_a, bound, ops.a_id, ops.b_id, ops.row_part,
                    ops.col_part)


def reference(ops, lo, hi):
    """-> transcripts (hi-lo, col_tiles, 16) u32 and {(row_tile, col_tile): jackpot} of every tile."""
    T = CpuRefBackend().transcripts(ops, lo, hi)
    J = R.jackpot_hashes(T, ops.seed_a)
    return T, {(lo + i, j): bytes(J[i, j]) for i, j in np.ndindex(J.shape[:2])}


def compare(so_full, so_small, ops, lo, hi, expected):
    """None if the library agrees with the reference on this case, else what differs (the first of it)."""
    res = so_full.search(ops, lo, hi)
    got = {(c.row_tile, c.col_tile): c.jackpot for c in res.candidates}
    if res.dropped or len(got) != len(res.candidates):
        return {"what": "saturated bound: count", "want": len(expected), "got": len(res.candidates),
                "dropped": res.dropped, "duplicates": len(res.candidates) - len(got)}, res
    for tile in sorted(expected):
        if got.get(tile) != expected[tile]:
            return {"what": "jackpot", "tile": tile, "want": expected[tile].hex(),
                    "got": got[tile].hex() if tile in got else None}, res
    if set(got) != set(expected):
        return {"what": "extra tiles", "tiles": sorted(set(got) - set(expected))[:10]}, res
    bound = sorted(R.le_int(j) for j in expected.values())[len(expected) // 2]
    want = {t: j for t, j in expected.items() if R.le_int(j) <= bound}
    small = so_small.search(with_bound(ops, bound), lo, hi)
    total = len(small.candidates) + small.dropped
    if total != len(want) or len(small.candidates) != min(len(want), so_small.cap):
        return {"what": "middle bound: count", "bound": f"{bound:064x}", "want": len(want), "got_count": total,
                "written": len(small.candidates), "cap": so_small.cap}, small
    for c in small.candidates:
        if want.get((c.row_tile, c.col_tile)) != c.jackpot:
            return {"what": "middle bound: candidate not in the reference set", "bound": f"{bound:064x}",
                    "tile": (c.row_tile, c.col_tile), "jackpot": c.jackpot.hex()}, small
    return None, res


def dump(dump_dir, desc, ops, lo, hi, T, expected, res, diff):
    dump_dir.mkdir(parents=True, exist_ok=True)
    base = dump_dir / f"fail-case{desc['case']}"
    tiles = sorted(expected)
    got = res.candidates if res is not None else []
    np.savez_compressed(base.with_suffix(".npz"), a_noised=ops.a_noised, bt_noised=ops.bt_noised,
                        seed_a=np.frombuffer(ops.seed_a, np.uint8), k=ops.cfg.k, r=ops.cfg.r,
                        rows_pattern=np.frombuffer(ops.cfg.rows.to_bytes(), np.uint8),
                        cols_pattern=np.frombuffer(ops.cfg.cols.to_bytes(), np.uint8), lo=lo, hi=hi,
                        want_transcripts=T, want_tiles=np.array(tiles, np.uint32).reshape(-1, 2),
                        want_jackpots=np.array([np.frombuffer(expected[t], np.uint8) for t in tiles]),
                        got_tiles=np.array([(c.row_tile, c.col_tile) for c in got], np.uint32).reshape(-1, 2),
                        got_jackpots=np.array([np.frombuffer(c.jackpot, np.uint8) for c in got]).reshape(-1, 32))
    base.with_suffix(".json").write_text(json.dumps({**desc, "lo": lo, "hi": hi, "diff": diff}, indent=1,
                                                    default=str))
    return str(base.with_suffix(".npz"))


def test1(so_path, tiles, seed, dump_dir, on_case=None, progress=sys.stderr, tile="8x16"):
    """`on_case(ops, lo, hi, expected)` runs before the library sees a case (tests: a stub fed the answers)."""
    so_full, so_small = SoBackend(so_path, cap=CAP), SoBackend(so_path, cap=SMALL_CAP)
    rng = np.random.default_rng(seed)
    t0, last, done, n_cases, by_k = time.time(), 0.0, 0, 0, dict.fromkeys(KS, 0)
    for desc, ops, lo, hi in cases(tiles, rng, tile):
        T, expected = reference(ops, lo, hi)
        if on_case:
            on_case(ops, lo, hi, expected)
        try:
            diff, res = compare(so_full, so_small, ops, lo, hi, expected)
        except KernelError as e:
            diff, res = {"what": "kernel error", "error": str(e), "code": e.code}, None
        if diff:
            return {"verdict": "FAIL", "case": {**desc, "lo": lo, "hi": hi}, "diff": diff,
                    "dump": dump(dump_dir, desc, ops, lo, hi, T, expected, res, diff),
                    "tiles_ok": done, "cases_ok": n_cases, "seconds": round(time.time() - t0, 1)}
        n_cases += 1
        done += len(expected)
        by_k[desc["k"]] += len(expected)
        if progress and time.time() - last >= 10:
            last = time.time()
            print(f"test1: {done}/{tiles} tiles, {n_cases} cases, {last - t0:.0f} s", file=progress, flush=True)
    return {"verdict": "PASS", "tile": tile, "tiles": done, "cases": n_cases, "tiles_by_k": by_k, "edges": list(EDGES),
            "seconds": round(time.time() - t0, 1)}


class Capture:
    """The pool of test 2: keeps what the miner submits."""
    url = "accept://local"
    connected = True

    def __init__(self):
        self.submits = []

    def submit(self, job_id, b64):
        self.submits.append(b64)
        return {"id": 1, "result": True, "error": None}


def test2(backend, shares, seed, k=2048, m=256, n=256, max_passes=40, tile="8x16"):
    try:
        import pearl_mining as pm
    except ImportError:
        pm = None
    rng = np.random.default_rng(seed)
    pool, stats = Capture(), Stats(backend.devices())
    mn = Miner(backend, pool, stats, k=k, m=m, n=n, portion_rows=64, nbits_override=EASY, tile=tile)
    hdr = header(rng)
    job = Job("accept", hdr, R.nbits_to_target(EASY), 1, 3, 1, time.time())
    mn.on_job(job)
    jw = mn.prepare_job(job)
    errors = lambda: sum(d["compute_errors"] for d in stats.dev.values())
    t0, found, passes = time.time(), 0, 0
    # every candidate of a portion is checked before stopping: a wrong kernel's candidate may be a real share by
    # luck (a tile passes the easy target with p ~ 1/64), the rest of its portion would not
    while len(pool.submits) < shares and passes < max_passes and not errors():
        pw = mn.prepare_pass(jw, passes, 1)
        passes += 1
        for lo in range(0, pw.ops.row_tiles, mn.portion_tiles):
            res = backend.search(pw.ops, lo, min(lo + mn.portion_tiles, pw.ops.row_tiles))
            for cand in res.candidates:
                found += 1
                mn.handle_share(pw, cand, res.device)
            if errors() or len(pool.submits) >= shares:
                break
    verified = []
    for b64 in pool.submits:
        if pm:
            ok, msg = pm.verify_plain_proof_for_cert_version(3, pm.IncompleteBlockHeader.from_bytes(hdr),
                                                             pm.PlainProof.from_base64(b64), nbits_override=EASY)
        else:
            import base64
            ok, msg = R.verify(hdr, R.PlainProof.from_bytes(base64.b64decode(b64)), nbits_override=EASY)
        verified.append({"ok": bool(ok), "msg": str(msg)})
    good = sum(v["ok"] for v in verified)
    verdict = "PASS" if good >= shares and errors() == 0 and good == len(verified) else "FAIL"
    out = {"verdict": verdict, "official": pm is not None, "shares_wanted": shares, "submitted": len(pool.submits),
           "verified_ok": good, "own_verify_failed": errors(), "candidates": found, "passes": passes,
           "k": k, "m": m, "n": n, "tile": tile, "nbits": f"{EASY:#010x}", "seconds": round(time.time() - t0, 1),
           "failures": [v["msg"] for v in verified if not v["ok"]][:5]}
    if pm is None:
        out["note"] = "pearl_mining not importable: shares checked by ref.verify only"
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="kernel library acceptance: tests 1 and 2 of TZ_v100_kernel.md §7")
    p.add_argument("--so", required=True, type=Path, help="the kernel library (pearl_search)")
    p.add_argument("--tile", choices=list(TILES), default="8x16", help="hash tile the library is built for")
    p.add_argument("--tiles", type=int, default=10 ** 6, help="test 1: tiles compared in all")
    p.add_argument("--shares", type=int, default=2, help="test 2: shares to find and verify")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("accept.json"))
    p.add_argument("--dump-dir", type=Path, default=Path("accept-dump"))
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    so = SoBackend(a.so)
    result = {"so": str(a.so.resolve()), "sha256": hashlib.sha256(a.so.read_bytes()).hexdigest(),
              "devices": so.devices(), "host": os.uname().nodename, "started": now(), "seed": a.seed, "tile": a.tile}
    result["test1"] = test1(a.so, a.tiles, a.seed, a.dump_dir, tile=a.tile)
    try:
        result["test2"] = test2(so, a.shares, a.seed, tile=a.tile)
    except KernelError as e:
        result["test2"] = {"verdict": "FAIL", "error": str(e), "code": e.code}
    verdicts = [result["test1"]["verdict"], result["test2"]["verdict"]]
    result["verdict"] = "FAIL" if "FAIL" in verdicts else "PASS"
    result["finished"] = now()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({"verdict": result["verdict"], "test1": result["test1"]["verdict"],
                      "test2": result["test2"]["verdict"], "out": str(a.out)}))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
