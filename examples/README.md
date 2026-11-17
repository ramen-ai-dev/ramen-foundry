# ramen-foundry examples

Runnable scripts for the four Core Architectures, plus the benchmarks and simulations built on them. Run everything from the repository root.

## Setup

```bash
git clone https://github.com/ramen-ai-dev/ramen-foundry.git
cd ramen-foundry
pip install -e .
export RAMEN_API_KEY="your-ramen-api-key"   # https://ramenai.dev/pricing
```

`RAMEN_API_KEY` is read from the environment, then from the checkout's `.env`, then from the file named by `RAMEN_ENV_FILE`. Set `OPENAI_API_KEY` as well to use OpenAI BYOK; otherwise ramen-ai uses managed-provider inference. The tools in every script are simulated: they record the call and change nothing.

## Core Architecture gate demos

Each demo runs one governed template through three stages:

| Stage | What happens | Needs a key |
|---|---|---|
| 1 Local | A deterministic pre-check blocks the call before any evaluation request | No |
| 2 Live | The ramen-ai bundle blocks a violating call; the BLOCK receipt is verified | Yes |
| 3 Live | The bundle allows a compliant call; the receipt is verified and the tool runs once | Yes |

Without `RAMEN_API_KEY` a demo runs Stage 1, prints how to enable Stages 2 and 3, and exits 0. With a key it runs all stages and exits 0 only when every stage ends as expected. Every live receipt is checked as Schema V5 Ed25519 against the production keys.

| Core Architecture | Script | Template and bundle | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|---|---|
| Financial & Authorization Gates | [`authorization_gate_demo.py`](authorization_gate_demo.py) | `AuthorizationGateAgent`, `ramen__financial_authorization_invariance` | $40k release with one approver | The requesting AI agent as an approver | Two independent approvals bound to amount, payee and account |
| Contractual & Eligibility Logic | [`eligibility_gate_demo.py`](eligibility_gate_demo.py) | `EligibilityGateAgent`, `ramen__contractual_eligibility_invariance` | Loss outside the policy term (host exclusion rule) | Exclusion shown only by the verified claim file | Full claim evidence, $63k approval |
| Compliance & Provenance Traceability | [`provenance_gate_demo.py`](provenance_gate_demo.py) | `ProvenanceGateAgent`, `ramen__compliance_provenance_invariance` | Caller-written `verifier_report`; tampered receipt | One evidence set for a batch of subjects | Receipts bound to subject, type and artifact; derived report |
| Cyber-Physical Kinematic Governance | [`kinematic_gate_demo.py`](kinematic_gate_demo.py) | `IndustrialAutomationAgent`, `ramen__industrial_iot_actuation_invariance` | NaN velocity, unknown action, undeclared key | Aerosol canister onto an active burner | De-rated collaborative bracket transfer |

```bash
python examples/authorization_gate_demo.py
python examples/eligibility_gate_demo.py
python examples/provenance_gate_demo.py
python examples/kinematic_gate_demo.py
```

The demos share [`_gate_showcase.py`](_gate_showcase.py): key loading, a spaced recording client, independent receipt verification and the summary table. Each demo exposes `run(client, live=True)` so you can drive it from your own code or tests.

The provenance demo first mints two evidence receipts by evaluating a benign `inspect_directory` call under `ramen__shield_core_it`. In production the receipts come from your own governed steps.

## Cyber-Physical benchmarks and simulations

| Script | What it does | Extras |
|---|---|---|
| [`benchmark_asimov_robotics.py`](benchmark_asimov_robotics.py) | Five ASIMOV and RoboHarm vectors (one allowed, four blocked) with a verified-receipt table | none |
| [`benchmark_roboharm_full_suite.py`](benchmark_roboharm_full_suite.py) | The full RoboHarm suite: five hazards and five safe controls | none |
| [`simulate_robotics_3d.py`](simulate_robotics_3d.py) | MuJoCo workcell: policy evaluation, mid-air kinetic arrest on a block | `pip install -e ".[simulation]"` |
| [`simulate_flagship_plunge.py`](simulate_flagship_plunge.py) | Bread versus infant mannequin cutlery simulation, optional recording | `pip install -e ".[simulation]"` |

## Other examples

| Script | What it does | Extras |
|---|---|---|
| [`benchmark_credit_data.py`](benchmark_credit_data.py) | German Credit benchmark: XGBoost, SHAP attribution and `CommercialLendingAgent` adverse actions | `pip install -e ".[credit-benchmark]"` |
| [`self_healing_agent_demo.py`](self_healing_agent_demo.py) | `RamenSteerNode` repair loop for a credit adverse-action notice, with episodic memory | none |
| [`demonstrate_collective_learning.py`](demonstrate_collective_learning.py) | Two agents share a repair through ramen forge | none |

## Tests

`pytest` runs the offline tests for every demo, including a drift guard that pins each demo payload to the payloads verified against the live bundles. To run all demos against the live API:

```bash
RAMEN_LIVE_TESTS=1 python -m pytest -m live tests/live/test_live_examples.py
```
