#!/usr/bin/env python3
"""Independent 15-second watchdog for one QTC E rental.

It starts before placement.  Once an order appears it guards parent liveness,
pilot/order ownership, elapsed cost, the hard deadline, and endpoint progress.
On any violation it uses the same public-action subprocess boundary to drop the
order without a ban and close the pilot row.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time

from qtc_rental import (ActionClient, current_order_snapshot, db_session, discover_pilot_order, fetch, iso_now,
                        read_json, atomic_json, verify_cleanup)


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value).astimezone(dt.timezone.utc)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def ownership_problem(conn: sqlite3.Connection, state: dict, grace_s: float) -> str | None:
    placed = state.get("placedAt")
    if not placed or (dt.datetime.now(dt.timezone.utc) - parse_time(placed)).total_seconds() < grace_s:
        return None
    server_id, order_id, pilot = state["serverId"], state.get("orderId"), state["pilot"]
    if not order_id:
        return "placed phase has no exact order id after ownership grace"
    pilot_row = conn.execute(
        "SELECT pilot FROM pilot_servers WHERE platform='clore' AND server_id=? AND valid_to IS NULL", (server_id,)
    ).fetchone()
    if pilot_row is None or pilot_row[0] != pilot:
        return f"pilot no longer owns server {server_id}"
    order_row = conn.execute(
        "SELECT 1 FROM order_versions WHERE platform='clore' AND server_id=? AND order_id=? AND valid_to IS NULL "
        "AND status IN ('active','paused')",
        (server_id, order_id),
    ).fetchone()
    if order_row is None:
        return f"exact order {order_id} is not open on server {server_id}"
    snap = current_order_snapshot(conn, order_id, server_id)
    if snap is not None:
        if snap.get("image") != state.get("image"):
            return f"order image changed to {snap.get('image')!r}"
        if str(snap.get("owner")) != str(state["expectedOwner"]):
            return f"order owner changed to {snap.get('owner')!r}"
        cards = snap.get("gpu_array") or []
        if len(cards) != 1 or "4090" not in str(cards[0]).lower():
            return f"order card composition changed to {cards!r}"
    return None


def abort(state_path: Path, db: Path, project_root: Path, reason: str) -> bool:
    out = state_path.parent
    state = read_json(state_path)
    atomic_json(out / "abort.json", {"at": iso_now(), "reason": reason, "state": state})
    bridge = Path(state.get("actionBridge") or (Path(__file__).resolve().parent / "action_bridge.py"))
    actions = ActionClient(bridge, project_root, state.get("python") or sys.executable)
    order_id = state.get("orderId")
    if not order_id:
        with db_session(db) as conn:
            order_id = discover_pilot_order(conn, state["serverId"], state["pilot"], state.get("armedAt"),
                                            state.get("image"))
    uncertain = state.get("placementOutcome") in ("pending", "uncertain", "possibly_accepted")
    if not order_id and state.get("phase") in ("placing", "placed", "running", "complete", "cleanup", "guarding"):
        # The action subprocess can outlive a suddenly killed coordinator.
        # Give collector enough time to expose the just-created order before
        # ending the pilot; otherwise a paid orphan could be hidden from us.
        until = time.monotonic() + float(state.get("discoveryTimeoutSeconds", 120))
        while time.monotonic() < until and not order_id:
            time.sleep(3)
            with db_session(db) as conn:
                order_id = discover_pilot_order(conn, state["serverId"], state["pilot"], state.get("armedAt"),
                                                state.get("image"))
    if order_id and not state.get("orderId"):
        state["orderId"] = order_id
        state.setdefault("placedAt", iso_now())
        atomic_json(state_path, state)
    if not order_id and uncertain:
        atomic_json(out / "watchdog-pending.json", {
            "at": iso_now(),
            "reason": reason,
            "state": state,
            "message": "placement outcome remains uncertain; pilot stays open and watchdog keeps searching",
        })
        return False
    cleanup = []
    if order_id:
        try:
            cleanup.append({"drop": actions.drop(state["serverId"], order_id)})
        except Exception as exc:
            cleanup.append({"dropError": str(exc)})
    end_ok = False
    try:
        cleanup.append({"end": actions.end(state["serverId"])})
        end_ok = True
    except Exception as exc:
        cleanup.append({"endError": str(exc)})
    verified = False
    if order_id:
        try:
            verify_cleanup(db, state["serverId"], order_id, timeout_s=float(state.get("verifyTimeoutSeconds", 180)))
            cleanup.append({"verified": "no open pilot/billable order/current order snapshot"})
            verified = True
        except Exception as exc:
            cleanup.append({"verifyError": str(exc)})
    atomic_json(out / "watchdog-cleanup.json", {"at": iso_now(), "reason": reason, "actions": cleanup})
    return verified if order_id else end_ok


def run(args) -> int:
    state_path = Path(args.state).resolve()
    db = Path(args.db).resolve()
    project_root = Path(args.project_root).resolve()
    last_progress = time.monotonic()
    last_log = None
    while True:
        state = read_json(state_path)
        phase = state.get("phase")
        if phase == "cleaned":
            return 0

        reason = None
        parent_pid = int(state["parentPid"])
        if not alive(parent_pid):
            reason = f"coordinator process {parent_pid} is gone"

        # If placement succeeded but the coordinator died before persisting the
        # returned id, collector/order_terms still give the exact pilot order.
        if not state.get("orderId") and phase in ("placing", "placed", "running", "complete", "cleanup", "guarding"):
            with db_session(db) as conn:
                found = discover_pilot_order(conn, state["serverId"], state["pilot"], state.get("armedAt"),
                                             state.get("image"))
            if found:
                state["orderId"] = found
                state.setdefault("placedAt", iso_now())
                atomic_json(state_path, state)

        billing_started = state.get("billingStartedAt") or state.get("placedAt")
        if billing_started:
            elapsed = (dt.datetime.now(dt.timezone.utc) - parse_time(billing_started)).total_seconds()
            estimated = float(state["creationFeeUsd"]) + float(state["priceUsdDayWithFee"]) * elapsed / 86400.0
            if elapsed >= float(state["deadlineSeconds"]):
                reason = reason or f"hard deadline exceeded ({elapsed:.1f}s)"
            if estimated > float(state["maxCostUsd"]):
                reason = reason or f"estimated cost exceeded (${estimated:.4f})"
            with db_session(db) as conn:
                reason = reason or ownership_problem(conn, state, args.ownership_grace)

        endpoint = state.get("endpoint")
        if endpoint and phase == "running":
            try:
                body = fetch(endpoint + "/run.log", timeout=8)
            except Exception:
                body = None
            if body is not None:
                mark = (len(body), hashlib.sha256(body).digest())
                if mark != last_log:
                    last_log = mark
                    last_progress = time.monotonic()
            endpoint_at = parse_time(state.get("endpointDiscoveredAt") or state["placedAt"])
            endpoint_age = (dt.datetime.now(dt.timezone.utc) - endpoint_at).total_seconds()
            if endpoint_age > args.startup_grace and time.monotonic() - last_progress > args.stall_seconds:
                reason = reason or f"run log/heartbeat stalled for >{args.stall_seconds}s"

        print(json.dumps({"at": iso_now(), "phase": phase, "orderId": state.get("orderId"), "reason": reason}), flush=True)
        if reason:
            if abort(state_path, db, project_root, reason):
                return 2
            if args.once:
                return 3
            time.sleep(args.interval)
            continue
        if args.once:
            return 0
        time.sleep(args.interval)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--project-root", required=True)
    p.add_argument("--interval", type=float, default=15.0)
    p.add_argument("--stall-seconds", type=float, default=75.0)
    p.add_argument("--startup-grace", type=float, default=45.0)
    p.add_argument("--ownership-grace", type=float, default=90.0)
    p.add_argument("--once", action="store_true")
    raise SystemExit(run(p.parse_args()))


if __name__ == "__main__":
    main()
