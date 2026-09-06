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

## Template catalogue

| Template | Public class | Bound policy scope | Consequential capabilities |
|---|---|---|---|
| `hrtech` | `ResumeScreeningAgent` | EU AI Act Annex III Proxy Bias Interceptor (`0d5ed2af-5e98-4a8c-92c3-dea26c07bf9a`) | Governed evidence-focused report; mandatory human review |
| `devbox-shield` | `DevboxShieldAgent` | `ramen__shield_core_it`: Destructive Execution, Infrastructure Abuse, Secret Exfiltration | Directory inspection, path deletion, process termination |
| `db-shield` | `DbShieldAgent` | `ramen__shield_core_it`: Destructive Execution and Infrastructure Abuse | Query/plan inspection, deadlock diagnosis, backend termination |
| `scout-shield` | `ScoutShieldAgent` | `ramen__shield_core_it`: OWASP ASI06 Indirect Prompt Injection and Secret Exfiltration | URL retrieval, extraction, approved local reads, publication |

`ramen__shield_core_it` is an immutable production bundle slug. The backend resolves it to the currently active policy UUIDs at request time; the signed receipt records the exact resolved UUIDs that ran. This lets policy implementations evolve without requiring client releases.

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
