"""Jev-compatible HTTP server for Laya.

Exposes TypeSafe AI's Jev evaluation API (https://docs.typesafe.ai/api) on top of
Laya, so code written against ``POST https://api.typesafe.ai/v1/systemone`` can
point at a self-hosted Laya instance by changing only the base URL and dropping
the API key.

Endpoints
---------
POST /v1/systemone
    Request  : {"model": str, "state": str|object|array, "questions": {...}}
    Response : {"model": str, "answers": {...},
                "usage": {"input_tokens": int, "output_tokens": int}}
GET  /v1/models
    The ``model`` ids this server accepts.
GET  /health
    Which checkpoints are resident.

``model`` semantics
-------------------
Jev requires ``model``. Here it selects the checkpoint:

* ``"jev-latest"`` / ``"jev-preview"`` / ``"jev-1.13.0"`` -> auto-route: Laya
  picks the best checkpoint (english/multilingual/typed-decisions) per request.
* ``"english"`` / ``"multilingual"`` / ``"typed-decisions"`` (and Laya's aliases)
  -> pin that checkpoint.

The response ``model`` reports the checkpoint that actually answered, so callers
can see the routing outcome (``multilingual`` for non-English text, and so on).

Laya's answers carry two fields Jev does not have -- an ``action`` (escalation
head) probability on every answer and a ``confidence`` on ``noul`` answers. Both
are stripped so the response is byte-for-byte the Jev shape.

Authentication
--------------
Pass ``--api-key <key>`` (or set ``LAYA_API_KEY``) to require
``Authorization: Bearer <key>`` on ``/v1/systemone`` and ``/v1/models``; a
missing or wrong key gets HTTP 401, matching Jev. ``/health`` stays open.

Install the server extras and run::

    pip install 'laya[server]'
    laya-serve --device cuda --preload --api-key "$LAYA_API_KEY"
"""
import os
import secrets
from typing import Any, Dict, List, Optional

from .router import DEFAULT_MODELS, Router, normalise_name

#: Jev model ids that mean "let Laya route automatically".
_JEV_AUTO_ROUTE = frozenset({"jev-latest", "jev-preview", "jev-1.13.0"})

#: Canonical checkpoint names plus the Jev aliases, advertised by GET /v1/models.
_KNOWN_MODELS = sorted(_JEV_AUTO_ROUTE | set(DEFAULT_MODELS))

_QTYPES = frozenset({"choice", "score", "noul"})


class ValidationError(ValueError):
    """A request body failed Jev's validation rules (surfaced as HTTP 422)."""


def resolve_model(model: Optional[str]) -> Optional[str]:
    """Map a Jev ``model`` field to a Laya checkpoint name, or None to auto-route.

    Jev model ids mean auto-route; Laya checkpoint names (``english`` /
    ``multilingual`` / ``typed-decisions`` and their aliases) pin a checkpoint.
    Raises :class:`ValidationError` for unknown ids.
    """
    if model is None:
        return None
    key = str(model).strip().lower()
    if key in _JEV_AUTO_ROUTE:
        return None
    try:
        return normalise_name(key)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def to_jev_answer(answer: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce one Laya answer to the exact Jev answer shape.

    Laya adds an ``action`` (escalation-head) probability to every answer and a
    ``confidence`` to ``noul`` answers; Jev has neither, so both are dropped.
    """
    qtype = answer.get("type")
    if qtype == "noul":
        return {"type": "noul", "noul": answer.get("noul")}
    if qtype == "choice":
        return {
            "type": "choice",
            "choice": answer.get("choice"),
            "probabilities": answer.get("probabilities"),
            "confidence": answer.get("confidence"),
        }
    if qtype == "score":
        return {
            "type": "score",
            "score": answer.get("score"),
            "legend": answer.get("legend"),
            "probabilities": answer.get("probabilities"),
            "confidence": answer.get("confidence"),
        }
    return dict(answer)


def to_jev_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Transform ``Router.predict`` output into the Jev response body."""
    routing = result.get("routing") or {}
    answers = {qid: to_jev_answer(ans) for qid, ans in result.get("answers", {}).items()}
    usage = result.get("usage") or {}
    return {
        "model": routing.get("model") or "laya",
        "answers": answers,
        "usage": {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        },
    }


def validate_questions(questions: Any) -> None:
    """Enforce Jev's question schema; raises :class:`ValidationError` on first problem."""
    if not isinstance(questions, dict) or not questions:
        raise ValidationError("'questions' must be a non-empty object")
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise ValidationError("question %r must be an object" % (qid,))
        qtype = q.get("type")
        if qtype not in _QTYPES:
            raise ValidationError("question %r has unknown type %r" % (qid, qtype))
        if "instructions" not in q:
            raise ValidationError("question %r is missing 'instructions'" % (qid,))
        if qtype == "choice":
            criteria = q.get("criteria")
            if not isinstance(criteria, dict) or not criteria:
                raise ValidationError(
                    "choice question %r requires a non-empty 'criteria' object" % (qid,)
                )
        elif qtype == "score":
            criteria = q.get("criteria")
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise ValidationError(
                    "score question %r requires a 'criteria' array of at least 2 levels" % (qid,)
                )


def build_router(
    models_dir: Optional[str] = None,
    device: Optional[str] = None,
    preload: bool = True,
) -> Router:
    """Build a Router, pointing at a local models dir when given, else the HF repos."""
    models = None
    if models_dir:
        models = {
            "english": (models_dir, None),
            "multilingual": (models_dir, "multilingual"),
            "typed-decisions": (models_dir, "typed-decisions"),
        }
    return Router(models=models, device=device, preload=preload)


def create_app(router: Router, api_key: Optional[str] = None):
    """Build the FastAPI app around a pre-built Router.

    When ``api_key`` is set, ``POST /v1/systemone`` and ``GET /v1/models``
    require ``Authorization: Bearer <api_key>`` (401 otherwise). ``/health``
    stays open for liveness probes.
    """
    try:
        from fastapi import FastAPI, HTTPException, Request
    except ImportError as exc:  # pragma: no cover - only without the server extra
        raise ImportError(
            "the Laya server needs FastAPI; install with `pip install 'laya[server]'`"
        ) from exc

    app = FastAPI(title="Laya (Jev-compatible)", version="0.1.0")

    def _check_auth(authorization: Optional[str]) -> None:
        if not api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="missing or invalid API key; use 'Authorization: Bearer <API_KEY>'",
            )
        token = authorization[len("Bearer "):].strip()
        if not secrets.compare_digest(token, api_key):
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.get("/health")
    def health():
        return {"status": "ok", "loaded": router.loaded}

    @app.get("/v1/models")
    def models(request: Request):
        _check_auth(request.headers.get("authorization"))
        return {
            "object": "list",
            "data": [{"id": mid, "object": "model"} for mid in _KNOWN_MODELS],
        }

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        _check_auth(request.headers.get("authorization"))
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=422, detail="invalid JSON body") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail="request body must be a JSON object")

        if "model" not in body or not isinstance(body["model"], str):
            raise HTTPException(status_code=422, detail="'model' (string) is required")
        if "state" not in body:
            raise HTTPException(status_code=422, detail="'state' is required")
        state = body["state"]
        if not isinstance(state, (str, dict, list)):
            raise HTTPException(status_code=422, detail="'state' must be a string, object, or array")

        try:
            checkpoint = resolve_model(body["model"])
            validate_questions(body.get("questions"))
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
            result = router.predict(state, body["questions"], model=checkpoint)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return to_jev_response(result)

    return app


def main(argv: Optional[List[str]] = None) -> None:
    """Console entry point (``laya-serve``)."""
    import argparse

    parser = argparse.ArgumentParser(description="Serve Laya behind a Jev-compatible HTTP API.")
    parser.add_argument("--host", default=os.environ.get("LAYA_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LAYA_PORT", "8000")))
    parser.add_argument("--device", default=os.environ.get("LAYA_DEVICE"))
    parser.add_argument("--models-dir", default=os.environ.get("LAYA_MODELS_DIR"))
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LAYA_API_KEY"),
        help="require 'Authorization: Bearer <key>' on the API endpoints (default: off)",
    )
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="preload all checkpoints at startup (default: on; use --no-preload to load lazily)",
    )
    args = parser.parse_args(argv)

    router = build_router(models_dir=args.models_dir, device=args.device, preload=args.preload)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the Laya server needs uvicorn; install with `pip install 'laya[server]'`"
        ) from exc
    uvicorn.run(create_app(router, api_key=args.api_key), host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
