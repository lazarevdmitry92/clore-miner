#!/usr/bin/env python3
"""Thin subprocess boundary around the project's public pilot actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PILOT = "qtc_own_miner"
BY = "qtc-e-rental"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    sub = p.add_subparsers(dest="command", required=True)
    place = sub.add_parser("place")
    place.add_argument("--server", required=True)
    place.add_argument("--currency", required=True)
    place.add_argument("--price", required=True, type=float)
    place.add_argument("--image", required=True)
    drop = sub.add_parser("drop")
    drop.add_argument("--server", required=True)
    drop.add_argument("--order", required=True)
    end = sub.add_parser("end")
    end.add_argument("--server", required=True)
    args = p.parse_args()

    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root))
    from common.env import load_dotenv
    load_dotenv()

    if args.command == "place":
        from ops import market_actions
        try:
            result = market_actions.place(
                args.server,
                "on-demand",
                args.currency,
                args.price,
                miner="srb",
                coin="qtc",
                pilot=PILOT,
                until="35m",
                exempt_from="all",
                image=args.image,
                by=BY,
            )
        except market_actions.ActionError as exc:
            print(json.dumps({
                "ok": False,
                "error": str(exc),
                "code": exc.code,
                "outcome": exc.outcome,
                "operationId": exc.operation_id,
                "requestSent": exc.request_sent,
                "possiblyAccepted": exc.possibly_accepted,
            }, ensure_ascii=False, sort_keys=True))
            raise SystemExit(2)
    elif args.command == "drop":
        from ops import drop as drop_ops
        try:
            result = drop_ops.drop("clore", args.server, BY, "QTC E rental calculation complete", False, args.order)
        except SystemExit as exc:
            # Cleanup is deliberately idempotent: the collector may already have
            # observed a watchdog/main-process drop.
            if "нет нашего заказа" not in str(exc):
                raise
            result = {"server_id": args.server, "order_id": args.order, "already_closed": True, "ban": False}
    else:
        from ops import pilots as pilot_ops
        from sources import store
        with store.session() as conn:
            closed = pilot_ops.end(conn, PILOT, BY, [str(args.server)])
        result = {"pilot": PILOT, "server_id": str(args.server), "closed": closed}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
