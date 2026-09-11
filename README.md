<p align="center">
  <a href="https://ramenai.dev">
    <img src="https://raw.githubusercontent.com/ramen-ai-dev/ramen-ai-integrations/master/assets/ramen-logo.png" alt="ramen-ai" width="100">
  </a>
</p>

<h1 align="center">ramen-foundry</h1>

<p align="center"><strong>Turnkey, legally governed AI agent templates powered by LangGraph and the ramen-ai stateless L2 execution boundary.</strong></p>

<p align="center">
  <a href="https://ramenai.dev">Platform</a> ·
  <a href="https://ramenai.dev/pricing">API keys</a> ·
  <a href="https://ramenai.dev/llms.txt">Architecture</a> ·
  <a href="https://github.com/ramen-ai-dev/ramen-ai-integrations">SDKs and integrations</a>
</p>

---

`ramen-foundry` is a Python library of policy-bound LangGraph components and agent templates. It places ramen-ai between model intent and consequential execution, then releases an action only when the semantic verdict allows it **and** the returned Ed25519 receipt verifies locally.

Use the low-level nodes to govern an existing graph, or start with a domain template:

- **hrtech** — evidence-focused resume review for a human decision-maker.
- **devbox-shield** — workstation inspection, cleanup, and process control.
- **db-shield** — database triage, query inspection, and deadlock diagnosis.
- **scout-shield** — injection-resistant web research and publication.

The package supplies governance boundaries and workflow structure; host applications retain ownership of credentials, model adapters, tools, infrastructure permissions, and human approvals. Policy enforcement supports compliance programs but does not by itself constitute legal advice or certification.

## Install

```bash
pip install ramen-foundry
```

Python 3.10 or newer is required. Obtain a ramen-ai API key at [ramenai.dev/pricing](https://ramenai.dev/pricing), then provide credentials through your environment or secret manager:

```bash
export RAMEN_API_KEY="your-ramen-api-key"
export OPENAI_API_KEY="your-provider-api-key"
```

Provider keys are required for bring-your-own-key inference unless your ramen-ai enterprise deployment supplies managed credentials. Supported provider routes are `openai`, `anthropic`, `google`, `synthetic`, and `hyperbolic`.

## Five-line quickstart

Assume `inspect_directory` is an application-defined LangChain `BaseTool` whose registered name is also `inspect_directory`:

```python
from os import environ
from ramen_ai import RamenClient
from ramen_foundry import DevboxShieldAgent, ToolInvocation
agent = DevboxShieldAgent(client=RamenClient(environ["RAMEN_API_KEY"]), tools={"inspect_directory": inspect_directory})
command = agent.execute(ToolInvocation(name="inspect_directory", arguments={"path": "./build"}, tool_call_id="inspect-1"))
```

`command` routes back to the configured LangGraph `llm_node` (default: `assistant`) with a `ToolMessage`, a cleared invocation, and either `governance_error=None` or an explicit denial/failure reason.

## Core engine architecture

![Core engine architecture: RamenToolNode evaluates and verifies each resolved action before a host capability can execute.](https://raw.githubusercontent.com/ramen-ai-dev/ramen-foundry/master/assets/core-engine-architecture.svg)

### `RamenToolNode`: pre-execution interception

`RamenToolNode` is the consequential-action boundary used by all three Shield agents. It:

1. Validates a resolved `ToolInvocation`.
2. Serializes `{"tool": name, "arguments": arguments}` as deterministic compact JSON.
3. Evaluates that payload against explicit policy UUIDs, stable bundle slugs, or both.
4. Requires both `allowed=True` and `receipt_verified=True`.
5. Invokes only a registered LangChain `BaseTool`.
6. Returns a LangGraph `Command` to the configured model/planner node.

Pre-execution failures—evaluation errors, blocked verdicts, missing or invalid receipts, and unknown tools—fail closed before a host capability is invoked. A host tool can still perform a partial side effect before raising an exception; that failure is reported explicitly but cannot be rolled back by Foundry. Safety-significant values must be explicit invocation arguments, and consequential tools should be idempotent or carry operation IDs so callers do not blindly retry an uncertain outcome. Hidden tool-side behavior cannot be semantically evaluated.

### `RamenGovernedNode`: self-correcting generation

![Governed generation architecture: content loops through semantic evaluation and one healing retry before verified release or blocking.](https://raw.githubusercontent.com/ramen-ai-dev/ramen-foundry/master/assets/governed-generation-architecture.svg)

`RamenGovernedNode` sends a prompt through the active governed-generation cascade. ramen-ai manages the provider call, semantic evaluation, and one healing retry. Only approved final content is written to graph state. Denials and transport/protocol failures produce `governed_content=None` and an explicit `governance_error`; blocked drafts are never released.

The Foundry node is synchronous and non-streaming. The underlying `ramen-ai-core` SDK also exposes streaming governed generation for applications that need progress events.

## Operational Scope & Boundary Demarcation

**The ingestion invariant.** ramen-foundry templates govern resolved tool-execution payloads at the graph's pre-execution boundary (`tools/pre-execute`). The L2 gate evaluates the proposed capability name and its explicit arguments against the bound policy or bundle, requires a locally verified receipt, and only then releases the registered host tool. It enforces invariant decision contracts on the payload presented to that boundary; it does not reconstruct facts that are absent from the payload.

**Upstream demarcation.** ramen-foundry does not parse raw credit-bureau files, perform raw-document OCR, clean source records, impute missing values, engineer model features, train underwriting models, or calculate SHAP values. The upstream application owns data licensing and provenance, extraction, normalization, missing-value treatment, feature engineering, model validation, protected-attribute controls, attribution-method selection, and the production mapping from model features to approved adverse-action reason codes. Foundry and ramen-ai do not make dirty or unsupported source evidence valid merely because it is submitted to a governed tool.

**Schema compliance.** The typed fields declared by each template are a caller contract, not an ingestion or coercion service. Callers must resolve safety-significant evidence into those fields before invoking the graph. The outer `ToolInvocation`, the bound semantic policy, local receipt verification, and the host tool's own argument schema form the combined pre-execution boundary: malformed dictionaries, omitted evidence, contradictory attribution, and unmapped free-form text must not be released as a successful tool execution and fail closed by design. Hosts must retain strict `BaseTool` schemas because Foundry does not transform dirty input into a valid typed payload.

### Empirical Verification: Real-World Credit Benchmark

The checked `examples/benchmark_credit_data.py` execution exercised the complete upstream-model-to-governed-tool boundary against production policies:

| Verification element | Empirical result |
|---|---|
| Dataset | OpenML `credit-g` v1: 1,000 rows, 20 features; SHA-256 `043dff5b02f794decc1540a561e63874e9a93717aa2e9d2b43ec56f3def7d68c` |
| Risk model | Deterministic XGBoost classifier; accuracy `0.752`, ROC-AUC `0.791` |
| Upstream TreeSHAP mapping | Principal adverse factors mapped to `INSUFFICIENT_LIQUIDITY`, `EXCESSIVE_REPAYMENT_TERM`, `INSUFFICIENT_EMPLOYMENT_HISTORY`, and `INSUFFICIENT_CASH_RESERVES` |
| Scenario A — grounded adverse action | **[ALLOWED]** with a locally verified Ed25519 receipt |
| Scenario B — hallucinated geographic factor | **[BLOCKED]** with a locally verified Ed25519 receipt and statutory steering |

This verifies the dual boundary on an authentic public credit distribution: grounded, model-attributable reasons can cross the execution boundary, while a reason that contradicts the SHAP evidence and introduces an unmapped geographic proxy fails closed. The benchmark remains an engineering verification, not a validated underwriting model or legal certification.

## Template catalogue

| Template | Public class | Bound policy scope | Consequential capabilities |
|---|---|---|---|
| `hrtech` | `ResumeScreeningAgent` | EU AI Act Annex III Proxy Bias Interceptor (`0d5ed2af-5e98-4a8c-92c3-dea26c07bf9a`) | Governed evidence-focused report; mandatory human review |
| `fintech` | `CommercialLendingAgent` | `ramen__fintech_banking_invariance`: adverse action (`796b7a87-d1f5-4ecc-91f2-a506a9b0d91e`) and wire dual control (`b4c18ba1-26b7-4b7f-b44b-65e8de790572`) | Credit adverse-action notices and commercial-loan wire disbursement |
| `devbox-shield` | `DevboxShieldAgent` | `ramen__shield_core_it`: Destructive Execution, Infrastructure Abuse, Secret Exfiltration | Directory inspection, path deletion, process termination |
| `db-shield` | `DbShieldAgent` | `ramen__shield_core_it`: Destructive Execution and Infrastructure Abuse | Query/plan inspection, deadlock diagnosis, backend termination |
| `scout-shield` | `ScoutShieldAgent` | `ramen__shield_core_it`: OWASP ASI06 Indirect Prompt Injection and Secret Exfiltration | URL retrieval, extraction, approved local reads, publication |

`ramen__shield_core_it` and `ramen__fintech_banking_invariance` are immutable production bundle slugs. The backend resolves each bundle to its currently active policy UUIDs at request time; the signed receipt records the exact resolved UUIDs that ran. This lets policy implementations evolve without requiring client releases.

### fintech commercial lending

`CommercialLendingAgent` governs a commercial-credit lifecycle against `ramen__fintech_banking_invariance`. The adverse-action control is anchored in ECOA Regulation B, CFPB Circular 2023-03, and FCRA § 615. The disbursement control enforces UCC § 4A-202 commercially reasonable security and dual control together with FinCEN Travel Rule evidence. These controls support a compliance program but do not replace counsel, bank procedures, sanctions screening, or authorized human approval.

| Tool | Required evidence parameters |
|---|---|
| `issue_credit_adverse_action` | `application_id`, `decision`, `reg_b_reason_codes`, `model_hash`, `shap_attribution_summary` |
| `dispatch_wire` | `account_id`, `amount_usd`, `beneficiary_name`, `beneficiary_routing`, `beneficiary_account`, `sanction_clearance_token`, `gl_offset`, `co_signer_public_key`, `co_signer_signature` |

Reason codes must be attributable to the identified underwriting model's negative feature evidence; unsupported or conversational reasons are blocked. High-value wires must carry machine-verifiable sanctions clearance and Ed25519 co-signer evidence rather than a conversational assertion. The host owns token issuance, key custody, signature creation, ledger authorization, and tool implementation; Foundry passes the explicit evidence unchanged through `RamenToolNode` for signed policy evaluation.

Five-line underwriter quickstart (the tool implementations are host-supplied LangChain `BaseTool` instances):

```python
from os import environ
from ramen_ai import RamenClient
from ramen_foundry import CommercialLendingAgent
agent = CommercialLendingAgent(client=RamenClient(environ["RAMEN_API_KEY"]), tools={"issue_credit_adverse_action": issue_credit_adverse_action, "dispatch_wire": dispatch_wire})
adverse_action, disbursement = agent.execute("issue_credit_adverse_action", rejection_evidence), agent.execute("dispatch_wire", signed_wire_evidence)
```

Use `agent.execute(tool_name, payload)` for a resolved direct action. Use `agent.invoke({"tool_invocation": invocation, "messages": []})` or compose `agent.graph` when the action is routed through the compiled LangGraph workflow. BYOK callers pass `provider_key` and `provider_name` together; enterprise managed-provider callers omit both.

#### Empirical German Credit benchmark

`examples/benchmark_credit_data.py` demonstrates the full responsibility boundary with the public UCI German Credit dataset exposed by OpenML as `credit-g` version 1. It performs all upstream work in the example application: protected/proxy feature exclusion, deterministic missing-value handling and one-hot encoding, XGBoost default-risk training, held-out metrics, real `shap.TreeExplainer` attribution, and an explicit source-feature-to-Regulation-B reason-code map. It then submits one grounded and one hallucinated adverse action through `CommercialLendingAgent`, requiring locally verified V5 Ed25519 receipts for both the live `[ALLOWED]` and `[BLOCKED]` outcomes.

The dataset is consumer-credit data and the mapping is illustrative; neither is a production commercial-underwriting model, validated adverse-action notice system, or legal conclusion. Institutions must substitute their approved data dictionary, model, attribution method, and principal-reason mapping. The script downloads public data and makes two live ramen-ai evaluations.

From a source checkout (the `examples/` directory is not installed by the wheel):

```bash
git clone https://github.com/ramen-ai-dev/ramen-foundry.git
cd ramen-foundry
python3 -m pip install -e '.[credit-benchmark]'
export RAMEN_API_KEY="your-ramen-api-key"
export OPENAI_API_KEY="your-provider-api-key"  # optional for enterprise managed mode
python3 examples/benchmark_credit_data.py
```

### hrtech

`ResumeScreeningAgent` compiles:

```text
START → draft_review_prompt → governed_resume_review → END
```

An application-supplied `BaseChatModel` drafts a neutral evidence-collection plan. `RamenGovernedNode` then generates the final report under the fixed Proxy Bias Interceptor. The result never represents a hiring, rejection, ranking, or eligibility decision and always returns `requires_human_review=True`.

```python
agent = ResumeScreeningAgent(
    llm=chat_model,
    client=client,
    provider_key=provider_key,
    provider_name="openai",
)
result = agent.screen(
    ResumeScreeningRequest(
        resume_text="Candidate resume text",
        job_description="Role requirements",
    )
)
```

The human-review flag is an application contract, not a built-in LangGraph interrupt or approval UI.

### devbox-shield

`DevboxShieldAgent` permits only these host-supplied tool names:

| Tool name | Intended capability |
|---|---|
| `inspect_directory` | Inspect paths, sizes, and cleanup candidates without mutation. |
| `delete_path` | Delete a host-approved cache, build output, or other path. |
| `terminate_process` | Terminate an explicitly identified orphan process. |

All requests are evaluated before tool lookup or execution. Attempts to remove system paths, user roots, shell configuration, credential material, or unrelated processes are expected to be denied by the bound Core IT controls. Hosts should additionally constrain deletion roots and process ownership inside their tool implementations.

### db-shield

`DbShieldAgent` permits:

| Tool name | Intended capability |
|---|---|
| `explain_query` | Run `EXPLAIN` through a read-only adapter. |
| `inspect_deadlocks` | Inspect lock graphs, blockers, and waiters. |
| `run_query` | Run a parameterized diagnostic/read query. |
| `terminate_backend` | Invoke a controlled `pg_terminate_backend` adapter. |

Query text and parameters are included in the governed payload. Destructive operations such as `DROP TABLE`, `TRUNCATE`, unscoped deletes, and unindexed bulk mutation requests can therefore be intercepted before database execution. Use least-privilege database roles, statement timeouts, transactions, and explicit environment identifiers as defence in depth.

### scout-shield

`ScoutShieldAgent` permits:

| Tool name | Intended capability |
|---|---|
| `fetch_url` | Retrieve an approved web resource. |
| `extract_content` | Parse or normalize retrieved material. |
| `read_local_file` | Read an explicitly approved research input. |
| `publish_research` | Publish an approved research artifact. |

Scraped pages, documents, and search results are untrusted input. Requests induced by embedded instructions—such as reading `.env`, collecting cloud credentials, curling secrets to an attacker, or publishing private data—are evaluated against OWASP ASI06 and secret-exfiltration controls before the capability can execute. Do not give research tools ambient access to secrets.

## Operational agent API

All three Shield agents share the same constructor and execution shape:

```python
ShieldAgent(
    *,
    client: RamenClient,
    tools: Mapping[str, BaseTool],
    llm_node: str = "assistant",
    provider_key: str | None = None,
    provider_name: str | None = None,
)

agent.execute(invocation: ToolInvocation | Mapping[str, Any]) -> Command
agent(state: Mapping[str, Any]) -> Command
```

- `execute(...)` is convenient for a resolved standalone action.
- `agent(state)` makes the instance a LangGraph node and expects `state["tool_invocation"]`.
- `tool_names` returns the sorted registered capability names.
- Registry keys must match each `BaseTool.name` and must belong to the template's documented capability set.
- The constructor always binds `bundle_ids=["ramen__shield_core_it"]`; callers cannot weaken or replace that scope.

### BYOK configuration

```python
agent = ScoutShieldAgent(
    client=RamenClient(os.environ["RAMEN_API_KEY"]),
    tools={"fetch_url": fetch_url},
    provider_key=os.environ["OPENAI_API_KEY"],
    provider_name="openai",
)
```

Pass `provider_key` and `provider_name` together. Omit both only when managed provider credentials are provisioned server-side.

## Security guarantees

### Stateless evaluation

Each evaluation contains the resolved action, explicit arguments, policy/bundle scope, and minimal context. ramen-ai does not need the agent's mutable LangGraph state to decide whether that action may cross the L2 boundary.

### Pre-execution interception

The governance call completes before registered capability lookup and invocation. A blocked action never reaches the host tool. This is materially different from output-only filtering after a shell command, SQL statement, or outbound request has already run.

### Fail-closed mechanics

A host tool is invoked only after all pre-execution conditions hold:

- The `ToolInvocation` is valid.
- The evaluation request succeeds.
- The semantic verdict is affirmative.
- The V5 receipt is present and cryptographically verified.
- The receipt's SHA-256 input binding matches the canonical action payload.
- The tool name is registered for that template.

For a valid invocation, governance and tool outcomes return an explicit `governance_error` on failure. Missing or malformed invocation state raises validation before evaluation and cannot execute a capability. Once an approved host tool begins, however, an exception may represent a partial side effect; inspect operation evidence before retrying.

### Ed25519 cryptographic receipts

`ramen-ai-core` verifies Ed25519 signatures and input hash binding locally. Receipts bind the verdict to the exact canonical input, resolved policy UUIDs, violations, statutory/control anchors, execution time, and outcome. Foundry requires `receipt_verified=True`; an unsigned or unverifiable allow is treated as a denial.

### Defence in depth

Semantic governance is not a replacement for OS permissions, sandboxing, read-only database roles, parameterized SQL, network egress controls, secret isolation, transaction boundaries, backups, human approvals, or application-specific allowlists. Keep those controls in place.

## Public exports

```python
from ramen_foundry import (
    DbShieldAgent,
    DevboxShieldAgent,
    EU_AI_ACT_PROXY_BIAS_POLICY_ID,
    RamenGovernedNode,
    RamenToolNode,
    ResumeScreeningAgent,
    ResumeScreeningRequest,
    ResumeScreeningResult,
    SHIELD_CORE_IT_BUNDLE_ID,
    ScoutShieldAgent,
    ToolInvocation,
)
```

`ToolInvocation` contains a non-empty `name`, an `arguments` dictionary, and a non-empty `tool_call_id`. `RamenToolNode` and `RamenGovernedNode` accept explicit `policy_ids`, `bundle_ids`, or both; at least one scope is required.

## Runtime dependencies

| Package | Constraint |
|---|---:|
| Python | `>=3.10` |
| `ramen-ai-core` | `>=0.3.2,<0.4.0` |
| `langgraph` | `==1.2.11` |
| `langchain-core` | `==1.6.0` |
| `pydantic` | `==2.13.4` |

## Resources

- [ramen-ai platform](https://ramenai.dev)
- [Plans and API keys](https://ramenai.dev/pricing)
- [Machine-readable architecture and integration context](https://ramenai.dev/llms.txt)
- [Python SDK and integrations](https://github.com/ramen-ai-dev/ramen-ai-integrations)
- [ramen-foundry source](https://github.com/ramen-ai-dev/ramen-foundry)

## License

MIT, as declared in `pyproject.toml`.
