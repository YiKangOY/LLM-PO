## 1) Define a TaskPlanning RAG knowledge corpus (planning-only, not command syntax)

### Why planning-only

Your `StrategyControllerAgent` step-2 already attaches stage command guidelines at runtime (timing/power/area) and generates the executable TCL command. Keeping **TaskPlanning RAG** focused on **strategy-level knowledge** avoids contaminating strategy descriptions with low-level flags/options. This matches your current TaskPlanning prompt constraint (“do not propose specific tool commands”) in `TaskPlanningAgent.plan_strategies()`.

### RAG corpus goals (must cover your 3 requirements)

1. **PPA tradeoff strategy planning** (timing vs power vs area sequencing, recovery concept).
2. **TraceSummary-aware diversification** (try new strategies, or similar strategy with different effort).
3. **Avoid list / avoid heuristics** (what patterns should be avoided and why).

### Recommended txt files (in your existing `docs/` directory)

Your RAG loader currently expects documents under:
`Path(__file__).parent.parent / "docs"`

Create these files:

#### File A — `docs/taskplanning_tradeoffs_and_templates.txt`

Include sections with **short, retrievable chunks**. Use `#` headers and `---` separators because your splitter uses those as separators.

Suggested structure (write content in plain text, *not code*):

* `# PPA Tradeoff Principles`

  * Timing improvement tends to increase area/power; plan recovery.
  * When to prioritize timing vs power vs area depending on gap-to-goal.
  * “Leave enough runtime for recovery” principles (from your target-selection strategy).

* `---`

* `# Strategy Templates (Round-level)`
  For each template, include:

  * **Name** (keyword-rich for retrieval)
  * **Optimization order** (e.g., `Timing → Power → Area` or `Timing → Area → Power`)
  * **Effort profile** (aggressive vs conservative; exploration vs exploitation)
  * **When to use**
  * **When to avoid**

  Examples:

  * Aggressive timing-first then recovery
  * Conservative timing-first then recovery
  * Timing stabilize (conservative) then recover power/area
  * Recovery-heavy (power/area) then resume timing (only if timing close but P/A overshot)

* `---`

* `# Effort Switching Rules (Explore vs Exploit)`

  * When budget sufficient: explore more targets/efforts early.
  * Later: exploit based on what worked + revealed constraints.
  * How to “try similar strategies with different effort” (e.g., aggressive → conservative variant).

#### File B — `docs/taskplanning_trace_summary_playbook.txt`

* `# How to Read Trace Memory Summary`

  * Identify **best outcome so far** vs **plateau**
  * Identify **tradeoff overshoot** (timing improved but power/area regressed too much)
  * Identify **strategy repetition** (same patterns repeated)

* `---`

* `# Diversify vs Refine Decision Rules`

  * If plateau: diversify (new order, new effort)
  * If improvement trend: refine (same order, adjust effort)
  * If power/area penalty too large: schedule recovery strategy next

* `---`

* `# Strategy Avoidance Heuristics from TraceSummary`

  * Avoid repeating “aggressive timing-first” if multiple traces show weak timing improvement and large P/A penalty.
  * Avoid repeated “conservative timing-only” if timing remains far from objective.
  * Avoid late-stage exploration if budget likely needed for recovery.

#### File C — `docs/taskplanning_avoid_rules.txt`

* `# Avoid Strategy Patterns`

  * A list of short “IF … THEN AVOID … BECAUSE …” rules.
  * Keep each rule short (1–3 sentences) so it becomes a single retrievable chunk.

> Keep your command syntax docs out of TaskPlanning RAG. Command syntax already lives in `COMMAND_GUIDELINES_BY_STAGE` and is injected by `StrategyControllerAgent.generate_command()` (step-2).

---

## 2) Update RAG loader to ingest these new txt files

Your current RAG database builder loads only `general_opt_strategies.txt` (and has unfix doc commented out).

### TODOs (rag_utils.py)

**File:** `rag_utils.py`

1. **Change `build_knowledge_database()` to load multiple planning docs**

   * Replace:

     * `strategy_file = docs_path / "general_opt_strategies.txt"`
   * With:

     * a list like:

       * `taskplanning_tradeoffs_and_templates.txt`
       * `taskplanning_trace_summary_playbook.txt`
       * `taskplanning_avoid_rules.txt`

2. **Create one Document per file**

   * Use metadata like:

     * `{"source": "taskplanning_tradeoffs_and_templates", "type": "task_planning"}`
     * `{"source": "taskplanning_trace_summary_playbook", "type": "task_planning"}`
     * `{"source": "taskplanning_avoid_rules", "type": "task_planning"}`

3. **Keep your existing splitter behavior**

   * You already split on `\n---\n` and `\n# ` to preserve semantic sections.
   * Ensure your new txt files use those separators intentionally.

4. **Add a “corpus version” to force rebuild when docs change**

   * Today, `load_database()` loads `vector_store.pkl` and will not rebuild if the underlying txt files change.
   * Add to the pickled metadata:

     * list of filenames
     * file modified timestamps (or hashes)
   * On `load_database()`, compare; if mismatch, rebuild.

**Acceptance criteria**

* `create_eco_rag_system()` rebuilds vector store automatically when you edit the planning txt files.

---

## 3) Extend TaskPlanning query generation to retrieve the right planning knowledge

Right now `TaskPlanningAgent.generate_planning_query()` produces a single query string.
To satisfy the 3 requirements reliably, make it produce **2–3 targeted queries**.

### TODOs (agentic_agents.py / TaskPlanningAgent)

**File:** `agentic_agents.py`

1. **Change the JSON output schema of `generate_planning_query()`**

   * From:

     ```json
     {"assessment": "...", "query": "..."}
     ```
   * To:

     ```json
     {
       "assessment": "...",
       "queries": [
         "query about tradeoff-aware strategy templates ...",
         "query about using TraceSummary to diversify/refine strategies ...",
         "query about which strategy patterns to avoid ..."
       ]
     }
     ```

2. **Update parsing logic**

   * Replace `_require_key(result, "query", ...)` with `_require_key(result, "queries", ...)`
   * Validate `queries` is a non-empty list of strings.

3. **Prompt changes for query generation (system prompt)**
   Add explicit instruction that queries must cover:

   * PPA tradeoff strategy planning
   * trace-summary-based diversification vs refinement
   * avoidance patterns

**Suggested updated system prompt (generate_planning_query)**

* Keep your current role, but change the “Task” section to:

  * “Generate exactly 3 queries: tradeoffs/templates, trace-summary diversification/refinement, avoid rules.”

4. **Use `retrieve_knowledge(queries, rag_system)`**

   * You already import `retrieve_knowledge` from `rag_helpers`.
   * It accepts a string or list and calls `rag_system.query_knowledge(k=1)` then formats text.

5. **Enable RAG in `plan_strategies()`**

   * You currently have retrieval commented out. Enable it:

     * `rag_content = retrieve_knowledge(queries, self.rag_system)`
   * Attach it in the user prompt as “RAG planning guidance”.

**Acceptance criteria**

* TaskPlanning prompt always sees:

  * objectives + current state + trace summary + retrieved planning guidance

---

## 4) Prompt changes for TaskPlanningAgent.plan_strategies() to use RAG + TraceSummary correctly

Your current `plan_strategies()` prompt already enforces:

* strategy descriptions include order (timing/power/area)
* explicit aggressiveness/conservativeness and Exploration/Exploitation labels
* no tool commands

### Add these behaviors (prompt-level)

#### A) Force “TraceSummary review” into the strategy generation step

Add an instruction block:

* “From Trace memory summary, identify:

  1. which high-level strategies have been tried (aggressive vs conservative; order patterns),
  2. which improved timing but hurt power/area,
  3. which produced little improvement (plateau).
     Then propose strategies that either diversify or refine.”

This directly supports your requirement (2). Trace summary is already passed in from framework via `TraceMemory.format_for_prompt()`.

#### B) Require at least one “variant effort” strategy when reusing a strategy family

Add a rule:

* “If you reuse a strategy family (e.g., timing-first), change the effort profile:

  * aggressive ↔ conservative OR exploration ↔ exploitation”

This ensures “similar strategies with different efforts.”

#### C) Require explicit “avoid” logic (without adding new output fields)

Since your output schema is fixed (description/commentary), embed avoid behavior in instructions:

* “Do not propose strategies that match the worst-performing trace patterns in TraceSummary unless you explicitly change the effort or order to address the failure mode.”

#### D) Feed RAG guidance as constraints

Add in user prompt:

* “RAG planning guidance (use as general rules/templates): …”

And in system prompt:

* “Use RAG guidance to select templates and avoidance heuristics; use TraceSummary as ground truth for what worked in this design.”

**Acceptance criteria**

* For `max_parallel=2`, you reliably get:

  * one exploratory strategy (new order or more aggressive)
  * one exploitative strategy (refine the most promising pattern)
* If TraceSummary indicates a strategy family failed, the proposed strategies shift away or change effort/order.

---

## 5) Ensure “strategies to avoid” are discoverable by RAG

This is primarily about **writing the docs** so retrieval works.

### TODOs (docs authoring)

1. In `taskplanning_avoid_rules.txt`, write rules using keywords the model will naturally query:

   * “aggressive timing-first”
   * “conservative timing-first”
   * “timing plateau”
   * “power/area overshoot”
   * “late-stage exploration”
   * “recovery scheduling”

2. In `taskplanning_trace_summary_playbook.txt`, include “mapping examples”:

   * “If timing improved but power/area increased too much → schedule recovery-first or conservative timing”
   * “If timing plateau → diversify target order or effort”
   * “If power/area already low but timing far → aggressive timing-first”

This ensures query hits return specific, actionable templates.

---

## 6) Optional: tighten RAG query + retrieval mechanics for TaskPlanning

### TODOs

1. Increase `k` to 2 for TaskPlanning only

* Today `retrieve_knowledge()` uses `k=1`.
* Add an optional parameter `k=2` so TaskPlanning can retrieve:

  * one template chunk + one avoid-rule chunk

2. Add query “prefixing” in the query generator prompt

* Ask the model to produce queries like:

  * “Strategy templates for timing-power-area order under tradeoffs …”
  * “Avoid rules for repeated aggressive timing-first when plateau …”
    This improves retrieval precision in small corpora.

---

## Summary of concrete implementation checklist

### Create/Update txt docs (docs/)

* [ ] `docs/taskplanning_tradeoffs_and_templates.txt` (tradeoffs + strategy templates + effort switching)
* [ ] `docs/taskplanning_trace_summary_playbook.txt` (how to interpret trace summary; diversify/refine)
* [ ] `docs/taskplanning_avoid_rules.txt` (avoid heuristics)

### Modify RAG system (rag_utils.py)

* [ ] Load multiple planning docs instead of single `general_opt_strategies.txt`
* [ ] Add corpus fingerprint/versioning to rebuild when docs change

### Modify TaskPlanningAgent (agentic_agents.py)

* [ ] `generate_planning_query()` returns `queries: [..]` (2–3 queries) instead of single `query`
* [ ] `plan_strategies()` enables `retrieve_knowledge(queries, rag_system)` and attaches RAG guidance into prompt
* [ ] Prompt updates: enforce trace-summary review, diversify/refine rules, avoid repetition unless effort/order changes
