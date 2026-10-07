# QTC E atlas-only rental runner

This is the coordinator-side package for the bounded QTC E compute run.  It
does not contain a platform client and was prepared/tested offline.  Copy this
directory to `atlas`; run it only from the repository checkout on `atlas`.

The default candidate is Clore `69333`, but it is never trusted by number.  On
every plan, and again after the independent watchdog is running, the runner
re-reads the collector database and requires:

- free server, owner `4727`, exactly one `rtx 4090`, advertised host CUDA
  compatibility at least 12.6, with the same owner/card count in `server_eval`;
- no owner 570, active ban of any scope, current order, or open pilot row;
- no current fleet margin above `$1/day` on either order type, and exact
  currency/native price/USD-price agreement between the live offer and the
  `server_eval` snapshot that produced that margin;
- current fee/rate data, price below `$2/hour`, and projected 35-minute cost
  (including creation fee) at or below `$1`;
- an immutable project image of the exact form
  `ghcr.io/lazarevdmitry92/clore-miner:qtcbench-cuda-<40hex>@sha256:<64hex>`.

## Dry-run plan (no platform request)

```sh
python3.11 /path/to/qtc-e-rental-runner/qtc_rental.py \
  --project-root "$PWD" \
  --server 69333 --expected-owner 4727 \
  --runner-inventory-sha256 19394edbaebb3de92d8d39c87ad0ca318f630d0aeb7bb68481aa58846b5cdb97 \
  --image 'ghcr.io/lazarevdmitry92/clore-miner:qtcbench-cuda-<commit>@sha256:<digest>' \
  --output /tmp/qtc-e-atlas-plan
```

This writes `/tmp/qtc-e-atlas-plan/plan.json` and exits.  It does not inspect
the registry or place an order.

## Confirmed paid run

Review `plan.json`, then repeat with a fresh output directory and `--yes`:

```sh
python3.11 /path/to/qtc-e-rental-runner/qtc_rental.py \
  --project-root "$PWD" \
  --server 69333 --expected-owner 4727 \
  --runner-inventory-sha256 19394edbaebb3de92d8d39c87ad0ca318f630d0aeb7bb68481aa58846b5cdb97 \
  --image 'ghcr.io/lazarevdmitry92/clore-miner:qtcbench-cuda-<commit>@sha256:<digest>' \
  --output /tmp/qtc-e-atlas-run-001 --yes
```

Both paths first verify `RUNNER_SHA256SUMS` against the exact digest shown
above and then verify the three executable Python files. The digest and file
inventory are written to plan/state, binding the atlas copy to reviewed code.
The confirmed path then resolves the unique image tag with `crane digest` and
requires the returned digest to match the supplied `@sha256`.  Because Clore's
placement API does not accept `tag@digest`, only the verified unique
`repo:qtcbench-cuda-<40hex>` tag is passed to `ops.market_actions.place`; the
expected and resolved digests remain bound in `plan.json` and `state.json`.
It starts `watchdog.py` as an independent process
before the final candidate recheck and placement.  Placement goes through
`ops.market_actions.place`, in `qtc_own_miner`, `until=35m`, `exempt=all`.
No SSH port is requested.

The endpoint is derived only after collector puts the returned order into
`current_items(source='orders-clore')`; the runner maps the internal port
21559 from that exact order's `pub_cluster` and `tcp_ports`.  It collects
`/done`, `/failed`, `/result.json`, `/parity.json`, `/run/manifest.json`,
`/ARTIFACTS.sha256`, and recursively mirrors `/run/` with SHA-256 inventory.
Every stable file named by the remote inventory is fetched through a strict
path allowlist and hash-verified. The runner then cross-checks binary,
reps=3/6 parity batches, parity artifact, manifest batches, and result before
accepting success. On `/failed`, it spends at most 45 seconds collecting the
failed marker, final logs, parity/manifest, and hash-bound partial batches
before teardown; failure evidence is never accepted as a successful result.

The 15-second watchdog independently enforces parent liveness, exact
order/pilot ownership, 30-minute deadline, `$1` elapsed-cost ceiling, stable
owner/card identity, and advancing `/run.log`.  A violation, success, error,
timeout, or interrupt all enter the same cleanup: `ops.drop` with no ban, then
`ops.pilots.end`.  Completion is accepted only after collector shows neither
an open pilot row nor a billable/open order/snapshot.

If Clore accepted a paid order but `market_actions.place` reports `lost`
before its id appears, cleanup does not close the pilot or stop the watchdog.
For up to 120 seconds it binds the collector order by server, pilot, exact
image tag, and placement-time window, then drops that recovered id and requires
the same collector convergence check.

The coordinator terminates the independent watchdog only after convergence is
proven. If coordinator cleanup fails, the watchdog remains in its own session;
on parent exit it makes the second exact-order drop/end attempt and records its
verification outcome instead of being reaped prematurely.
The coordinator also polls watchdog liveness throughout placement, endpoint
discovery, and the run; an unexpected watchdog exit is itself fail-closed.
The 30-minute elapsed/cost clock starts at `armedAt`, before placement, so API
or collector-discovery delay cannot extend paid time.

## Offline checks

```sh
python3.11 -m py_compile \
  /path/to/qtc-e-rental-runner/qtc_rental.py \
  /path/to/qtc-e-rental-runner/action_bridge.py \
  /path/to/qtc-e-rental-runner/watchdog.py
python3.11 -m pytest -q /path/to/qtc-e-rental-runner/tests
```

Tests use a temporary SQLite database and a mocked action subprocess.  They do
not access Clore, the image registry, or `atlas`.
