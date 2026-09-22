"""Tests for the Jev-compatible server mapping. Pure functions only: no model
weights are loaded and FastAPI is not required (it is imported lazily)."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.server import (  # noqa: E402
    ValidationError,
    resolve_model,
    to_jev_answer,
    to_jev_response,
    validate_questions,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def raises(name, fn, exc_type):
    try:
        fn()
    except exc_type:
        PASS.append(name)
    except Exception as e:  # noqa: BLE001
        FAIL.append("%s: got %r, want %s" % (name, e, exc_type.__name__))
    else:
        FAIL.append("%s: no exception raised, want %s" % (name, exc_type.__name__))


# --------------------------------------------------------------------- model resolution
check("resolve: jev-latest -> auto-route", resolve_model("jev-latest"), None)
check("resolve: jev-preview -> auto-route", resolve_model("jev-preview"), None)
check("resolve: jev-1.13.0 -> auto-route", resolve_model("jev-1.13.0"), None)
check("resolve: jev-latest is case-insensitive", resolve_model("JEV-LATEST"), None)
check("resolve: english pins", resolve_model("english"), "english")
check("resolve: multilingual pins", resolve_model("multilingual"), "multilingual")
check("resolve: typed-decisions pins", resolve_model("typed-decisions"), "typed-decisions")
check("resolve: alias 'laya' -> english", resolve_model("laya"), "english")
check("resolve: alias 'ml' -> multilingual", resolve_model("ml"), "multilingual")
raises("resolve: unknown model rejected", lambda: resolve_model("gpt-4"), ValidationError)
raises("resolve: jev-1.13 (typo) rejected", lambda: resolve_model("jev-1.13"), ValidationError)

# --------------------------------------------------------------------- answer mapping
check(
    "answer: noul drops confidence + action",
    to_jev_answer({"type": "noul", "noul": 0.82, "confidence": 0.82, "action": {"act_probability": 1.0}}),
    {"type": "noul", "noul": 0.82},
)
check(
    "answer: choice drops action",
    to_jev_answer(
        {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9, "technical": 0.1},
            "confidence": 0.86,
            "action": {"act_probability": 1.0},
        }
    ),
    {
        "type": "choice",
        "choice": "billing",
        "probabilities": {"billing": 0.9, "technical": 0.1},
        "confidence": 0.86,
    },
)
check(
    "answer: score keeps legend + probabilities + confidence, drops action",
    to_jev_answer(
        {
            "type": "score",
            "score": 1.44,
            "legend": {"0": "not urgent", "1": "soon", "2": "critical"},
            "probabilities": {"0": 0.12, "1": 0.32, "2": 0.56},
            "confidence": 0.14,
            "action": {"act_probability": 1.0},
        }
    ),
    {
        "type": "score",
        "score": 1.44,
        "legend": {"0": "not urgent", "1": "soon", "2": "critical"},
        "probabilities": {"0": 0.12, "1": 0.32, "2": 0.56},
        "confidence": 0.14,
    },
)

# --------------------------------------------------------------------- response mapping
check(
    "response: Router.predict payload -> Jev shape",
    to_jev_response(
        {
            "model": "laya-rl-agent",
            "answers": {
                "is_urgent": {"type": "noul", "noul": 0.95, "confidence": 0.95, "action": {"act_probability": 1.0}},
                "department": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {"billing": 0.88, "technical": 0.12},
                    "confidence": 0.81,
                    "action": {"act_probability": 1.0},
                },
            },
            "usage": {"input_tokens": 296, "output_tokens": 0},
            "routing": {"model": "multilingual", "reason": "non-Latin script", "repo": "convaiinnovations/laya"},
        }
    ),
    {
        "model": "multilingual",
        "answers": {
            "is_urgent": {"type": "noul", "noul": 0.95},
            "department": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.88, "technical": 0.12},
                "confidence": 0.81,
            },
        },
        "usage": {"input_tokens": 296, "output_tokens": 0},
    },
)
check(
    "response: missing routing falls back to 'laya'",
    to_jev_response({"answers": {}, "usage": {}})["model"],
    "laya",
)

# --------------------------------------------------------------------- question validation
GOOD = {
    "a": {"type": "noul", "instructions": "yes or no?"},
    "b": {"type": "choice", "instructions": "pick", "criteria": {"x": "one", "y": "two"}},
    "c": {"type": "score", "instructions": "rate", "criteria": ["low", "mid", "high"]},
}
check("validate: mixed valid questions pass", validate_questions(GOOD) is None, True)
raises("validate: empty questions rejected", lambda: validate_questions({}), ValidationError)
raises("validate: non-object questions rejected", lambda: validate_questions("nope"), ValidationError)
raises("validate: unknown type rejected", lambda: validate_questions({"a": {"type": "boolean"}}), ValidationError)
raises(
    "validate: choice without criteria rejected",
    lambda: validate_questions({"a": {"type": "choice", "instructions": "pick"}}),
    ValidationError,
)
raises(
    "validate: score with one level rejected",
    lambda: validate_questions({"a": {"type": "score", "instructions": "rate", "criteria": ["only"]}}),
    ValidationError,
)
raises(
    "validate: missing instructions rejected",
    lambda: validate_questions({"a": {"type": "noul"}}),
    ValidationError,
)

# --------------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all server-mapping tests passed")
sys.exit(1 if FAIL else 0)
