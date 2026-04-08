"""
Inference Script — Legal Document Review OpenEnv
=================================================
Runs an LLM agent against all 3 tasks and emits structured logs.

MANDATORY VARIABLES:
    API_BASE_URL    LLM endpoint
    MODEL_NAME      Model identifier  
    HF_TOKEN        API key (NO default — read from env only)

STDOUT FORMAT (strictly):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# MANDATORY: Environment variables — defaults only for API_BASE_URL, MODEL_NAME
# HF_TOKEN has NO default (per spec)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default — must be set in environment

# Optional — only used if running via docker image
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ---------------------------------------------------------------------------
# Other configuration
# ---------------------------------------------------------------------------
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "legal_review"

TASK_CONFIGS = {
    "nda_review": {
        "max_steps": 14,
        "success_threshold": 0.50,
        "temperature": 0.3,
    },
    "service_agreement_review": {
        "max_steps": 18,
        "success_threshold": 0.45,
        "temperature": 0.3,
    },
    "merger_agreement_review": {
        "max_steps": 22,
        "success_threshold": 0.40,
        "temperature": 0.4,
    },
}


# ---------------------------------------------------------------------------
# Strict stdout logging — [START] / [STEP] / [END]
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    """Emit exactly one [START] line at episode begin."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """Emit one [STEP] line immediately after env.step() returns."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Keep action on single line — no embedded newlines
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    """Emit exactly one [END] line after episode completes (always emitted)."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------

class LegalReviewClient:
    """Thin HTTP client wrapping the Legal Review environment REST API."""

    def __init__(self, base_url: str = ENV_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, task_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}/step",
            json={"task_id": task_id, "action": action},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def grade(self, task_id: str) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}/grade",
            params={"task_id": task_id},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# System prompt for the LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert legal counsel reviewing contracts and legal documents.
Your task is to thoroughly review the document by taking structured actions.

You must respond with a single valid JSON object representing one action.
Available action types and required fields:

1. READ_SECTION:       {"action_type": "read_section", "section_id": "<s1|s2|...>"}
2. SEARCH_DOCUMENT:    {"action_type": "search_document", "query": "<search text>"}
3. IDENTIFY_CLAUSE:    {"action_type": "identify_clause", "section_id": "<id>",
                        "clause_type": "<type>", "clause_excerpt": "<text>", "confidence": 0.9}
   Clause types: indemnification, limitation_of_liability, confidentiality, termination,
                 governing_law, dispute_resolution, payment_terms, intellectual_property,
                 non_compete, force_majeure, representations_warranties, assignment
4. FLAG_ISSUE:         {"action_type": "flag_issue", "section_id": "<id>",
                        "issue_description": "<description>",
                        "risk_level": "<low|medium|high|critical>",
                        "suggested_revision": "<optional revision text>"}
5. ASSESS_RISK:        {"action_type": "assess_risk", "risk_level": "<low|medium|high|critical>"}
6. EXTRACT_PARTY:      {"action_type": "extract_party", "party_name": "<full name>"}
7. EXTRACT_DATE:       {"action_type": "extract_date", "date_value": "<date>",
                        "date_context": "<what this date represents>"}
8. EXTRACT_OBLIGATION: {"action_type": "extract_obligation",
                        "obligation_description": "<description>"}
9. SUMMARIZE_DOCUMENT: {"action_type": "summarize_document",
                        "summary_text": "<comprehensive summary, 100+ words>"}
10. RECOMMEND_REVISION:{"action_type": "recommend_revision", "section_id": "<id>",
                        "suggested_revision": "<revision text>"}
11. SUBMIT_REVIEW:     {"action_type": "submit_review",
                        "overall_risk": "<low|medium|high|critical>",
                        "review_notes": "<final review notes>"}

Strategy for high scores:
- Read ALL sections before drawing conclusions (READ_SECTION for each)
- Identify the major legal clause types present (IDENTIFY_CLAUSE)
- Extract all contracting parties and key dates
- Flag problematic provisions with appropriate risk level (FLAG_ISSUE)
- Set overall risk assessment (ASSESS_RISK)  
- Write a comprehensive summary mentioning key risks (SUMMARIZE_DOCUMENT)
- Submit your review when done (SUBMIT_REVIEW)

Respond with ONLY the JSON object. No explanation, no markdown, no code blocks.
""").strip()


def build_user_prompt(
    obs: Dict[str, Any],
    step: int,
    history: List[str],
) -> str:
    """Build a context-rich user prompt from the current observation."""
    current_section = obs.get("current_section")
    sections_read = obs.get("sections_read", [])
    clauses = obs.get("identified_clauses", [])
    issues = obs.get("flagged_issues", [])
    parties = obs.get("parties_extracted", [])
    dates = obs.get("key_dates_extracted", [])
    obligations = obs.get("obligations_extracted", [])
    hints = obs.get("progress_hints", [])
    total_sections = obs.get("total_sections", 0)
    max_steps = obs.get("max_steps", 20)

    all_section_ids = [f"s{i + 1}" for i in range(total_sections)]
    unread = [s for s in all_section_ids if s not in sections_read]

    current_section_text = ""
    if current_section:
        content = current_section.get("content", "")
        current_section_text = (
            f"\nCURRENT SECTION [{current_section.get('section_id')}]"
            f" — {current_section.get('title')}:\n{content}\n"
        )

    history_block = "\n".join(history[-5:]) if history else "None"

    clause_summary = [
        f"{c['clause_type']} in {c['section_id']}" for c in clauses
    ]
    issue_summary = [
        f"{i['risk_level']}: {i['description'][:60]}" for i in issues
    ]

    return textwrap.dedent(f"""
        DOCUMENT: {obs.get('document_title', 'Unknown')}
        TASK: {obs.get('task_description', '')[:300]}

        STEP: {step} / {max_steps}
        LAST RESULT: {obs.get('last_action_result', '')[:300]}

        PROGRESS:
        - Sections read ({len(sections_read)}/{total_sections}): {sections_read}
        - Unread sections: {unread}
        - Clauses identified: {clause_summary}
        - Issues flagged ({len(issues)}): {issue_summary}
        - Parties extracted: {parties}
        - Dates extracted: {dates}
        - Obligations extracted: {len(obligations)}
        - Risk assessment: {obs.get('risk_assessment', 'NOT SET')}
        - Summary submitted: {'YES' if obs.get('document_summary') else 'NO'}
        - Review complete: {obs.get('review_complete', False)}
        {current_section_text}
        HINTS: {hints}

        RECENT ACTIONS:
        {history_block}

        What is your next action? Respond with JSON only:
    """).strip()


# ---------------------------------------------------------------------------
# LLM call via OpenAI client (mandatory — uses API_BASE_URL + HF_TOKEN)
# ---------------------------------------------------------------------------


def get_llm_action(
    client: OpenAI,
    obs: Dict[str, Any],
    step: int,
    history: List[str],
    temperature: float = 0.3,
) -> Tuple[Dict[str, Any], str]:
    """
    Call the LLM via OpenAI client and parse the returned action.
    Returns (action_dict, action_str_for_logging).
    """
    user_prompt = build_user_prompt(obs, step, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=400,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        action_dict = json.loads(text)
        action_str = json.dumps(action_dict)
        return action_dict, action_str

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        # Fallback: read the next unread section, or submit
        sections_read = obs.get("sections_read", [])
        total = obs.get("total_sections", 1)
        for i in range(1, total + 1):
            sid = f"s{i}"
            if sid not in sections_read:
                fallback = {"action_type": "read_section", "section_id": sid}
                return fallback, json.dumps(fallback)
        # All sections read — submit
        fallback = {
            "action_type": "submit_review",
            "overall_risk": "medium",
            "review_notes": "Review complete based on document analysis.",
        }
        return fallback, json.dumps(fallback)

    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        fallback = {"action_type": "read_section", "section_id": "s1"}
        return fallback, json.dumps(fallback)


# ---------------------------------------------------------------------------
# Run a single task episode
# ---------------------------------------------------------------------------

def run_task(
    client: OpenAI,
    env_client: LegalReviewClient,
    task_id: str,
) -> Tuple[bool, int, float, List[float]]:
    """
    Run one full episode for a task.
    Returns (success, steps_taken, final_score, rewards_list).
    Emits [START], [STEP]*, [END] to stdout.
    """
    config = TASK_CONFIGS[task_id]
    max_steps: int = config["max_steps"]
    success_threshold: float = config["success_threshold"]
    temperature: float = config["temperature"]

    rewards: List[float] = []
    history: List[str] = []
    steps_taken: int = 0
    final_score: float = 0.0
    success: bool = False

    # --- [START] ---
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset the environment
        result = env_client.reset(task_id=task_id)
        obs = result.get("observation", {})
        done = result.get("done", False)

        for step in range(1, max_steps + 1):
            if done:
                break

            # Get action from LLM (via OpenAI client)
            action_dict, action_str = get_llm_action(
                client, obs, step, history, temperature=temperature
            )

            # Execute action in environment
            error_msg: Optional[str] = None
            try:
                step_result = env_client.step(task_id=task_id, action=action_dict)
                new_obs = step_result.get("observation", {})
                reward = float(step_result.get("reward", 0.0))
                done = step_result.get("done", False)

                # Surface action errors from the environment
                last_result = new_obs.get("last_action_result", "")
                if last_result.startswith("ERROR:"):
                    error_msg = last_result.replace("\n", " ")[:80]

            except requests.HTTPError as http_err:
                reward = 0.0
                done = False
                error_msg = str(http_err)[:80]
                new_obs = obs  # keep previous observation

            rewards.append(reward)
            steps_taken = step
            obs = new_obs

            # --- [STEP] --- (one per step, immediately after step())
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error_msg,
            )

            history.append(
                f"Step {step}: {action_dict.get('action_type', '?')} -> "
                f"reward={reward:.2f}"
            )

            if done:
                break

            time.sleep(0.1)  # Be polite to the server

        # Get final graded score
        try:
            grade_result = env_client.grade(task_id=task_id)
            final_score = float(grade_result.get("score", 0.0))
        except Exception:
            # Fallback: normalise cumulative reward
            final_score = min(
                sum(r for r in rewards if r > 0) / max(max_steps * 0.3, 1.0),
                1.0,
            )

        final_score = round(min(max(final_score, 0.0), 1.0), 3)
        success = final_score >= success_threshold

    except Exception as e:
        print(f"[DEBUG] Unhandled error in task {task_id}: {e}", flush=True)
        success = False

    finally:
        # --- [END] --- always emitted, even on exception
        log_end(
            success=success,
            steps=steps_taken,
            score=final_score,
            rewards=rewards,
        )

    return success, steps_taken, final_score, rewards


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Build the OpenAI client — mandatory, uses env vars
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,  # HF_TOKEN, no default
    )
    env_client = LegalReviewClient(base_url=ENV_BASE_URL)

    # Wait for the environment server to be ready (up to 20 s)
    print("[DEBUG] Waiting for environment server...", flush=True)
    for attempt in range(10):
        if env_client.health():
            print("[DEBUG] Server ready.", flush=True)
            break
        time.sleep(2)
    else:
        print("[DEBUG] Server not available after 10 attempts. Exiting.", flush=True)
        sys.exit(1)

    # Determine which tasks to run
    task_arg = os.getenv("TASK_NAME", "")
    if task_arg and task_arg in TASK_CONFIGS:
        tasks_to_run = [task_arg]
    else:
        tasks_to_run = list(TASK_CONFIGS.keys())

    # Run each task
    all_results: Dict[str, Dict[str, Any]] = {}
    for task_id in tasks_to_run:
        print(f"\n{'=' * 60}", flush=True)
        print(f"[DEBUG] Starting task: {task_id}", flush=True)
        print(f"{'=' * 60}", flush=True)

        success, steps, score, rewards = run_task(client, env_client, task_id)
        all_results[task_id] = {
            "success": success,
            "steps": steps,
            "score": score,
            "rewards": rewards,
        }
        time.sleep(1)

    # Print summary (to stderr so it doesn't interfere with structured stdout)
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("BASELINE SUMMARY", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    for task_id, result in all_results.items():
        status = "PASS" if result["success"] else "FAIL"
        print(
            f"{status} | {task_id:40s} | score={result['score']:.3f} | steps={result['steps']}",
            file=sys.stderr,
        )

    if all_results:
        avg = sum(r["score"] for r in all_results.values()) / len(all_results)
        print(f"\nOverall average score: {avg:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
