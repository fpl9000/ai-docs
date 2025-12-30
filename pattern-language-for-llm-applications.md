# A Pattern Language for LLM Applications

## A Proposal for Documenting Recurring Solutions in LLM System Design

---

**Inspired by:** [@antiali.as on BlueSky](https://bsky.app/profile/antiali.as/post/3mb6ar6echs2k), who observed that "Someone's gotta start compiling a Pattern Language for LLM applications."

**With additional patterns contributed by:** @antiali.as, including Prefilling, Rollout Evaluation, Prompt Self-Improvement, Context Ablation, Test-Based Grounding, Skills, and In-Context Transfer Learning.

**In the tradition of:** Christopher Alexander's *A Pattern Language* (1977), which documented 253 architectural patterns forming a coherent language for building human-centered spaces—and which subsequently inspired the software design patterns movement.

---

## Preface

Christopher Alexander's insight was profound: patterns are not isolated solutions but form a *language*—a network of interrelated solutions that reference and reinforce each other. Just as words gain meaning through their relationships to other words, design patterns gain power through their connections to other patterns.

The field of LLM application development has, over the past few years, developed its own recurring solutions to recurring problems. These patterns have emerged independently across teams building chatbots, coding assistants, autonomous agents, and enterprise AI systems. Yet they remain largely undocumented as a coherent language.

This proposal aims to begin that documentation. Each pattern follows Alexander's format:

1. **A name** (functioning as a vocabulary term)
2. **A problem statement** (the recurring challenge)
3. **The context** (when this problem arises)
4. **The solution** (the proven approach)
5. **Related patterns** (connections to other patterns in the language)

The patterns are organized roughly from foundational (dealing with basic LLM interaction) to emergent (dealing with complex multi-agent systems).

---

## Part I: Context and Grounding Patterns

These patterns address the fundamental challenge: LLMs are trained on static data but must operate in dynamic, context-specific situations.

---

### Pattern 1: Preloaded Context

**Problem:** An LLM has vast general knowledge but lacks awareness of the specific domain, user, or situation it must address in this conversation.

**Context:** You are building an application where the LLM needs domain-specific knowledge, user preferences, organizational policies, or situational awareness that wasn't present in its training data.

**Solution:** Inject relevant context into the system prompt or early in the conversation, *before* the user's query is processed. This preloaded context acts as the LLM's "working memory" for the session.

The preloaded context typically includes:

- **Identity and role definition** ("You are a customer support agent for Acme Corp...")
- **Domain knowledge** (product catalogs, policy documents, technical specifications)
- **User profile data** (preferences, history, permissions)
- **Temporal context** (current date, recent events, deadlines)
- **Behavioral constraints** (tone, forbidden topics, required disclaimers)

The key insight is that context should be *preloaded* rather than retrieved on-demand for information that is always relevant. This reduces latency and ensures consistent behavior from the first interaction.

**Therefore:** Structure your system prompt as a layered document: identity first, then domain knowledge, then user-specific context, then behavioral guidelines. Keep the total context within the model's effective attention window—typically the first 4,000–8,000 tokens receive the most reliable attention.

**Related Patterns:** (2) Retrieval-Augmented Generation for dynamic context; (4) System Prompt Architecture for structuring preloaded context; (11) Memory Layer for persisting context across sessions.

---

### Pattern 2: Retrieval-Augmented Generation (RAG)

**Problem:** The information needed to answer a query is too vast to preload, changes frequently, or depends on the specific question being asked.

**Context:** You have a knowledge base (documents, databases, APIs) that contains information the LLM needs, but this information is too large to fit in context, too dynamic to embed in prompts, or too query-dependent to preload.

**Solution:** Implement a retrieval step that, given the user's query, fetches relevant information from external sources and injects it into the LLM's context just-in-time.

The canonical RAG pipeline:

1. **Index:** Chunk documents and create embeddings stored in a vector database
2. **Retrieve:** Given a query, find semantically similar chunks
3. **Augment:** Insert retrieved chunks into the prompt as context
4. **Generate:** Have the LLM generate a response grounded in the retrieved context

Advanced RAG variations include:

- **HyDE (Hypothetical Document Embeddings):** Generate a hypothetical answer first, then use it to retrieve similar documents
- **Multi-query RAG:** Reformulate the query multiple ways to improve recall
- **Agentic RAG:** Let the LLM decide when and what to retrieve iteratively

**Therefore:** When building a RAG system, invest heavily in the retrieval quality. The best LLM cannot overcome poor retrieval. Chunk documents thoughtfully (semantic boundaries, not arbitrary token counts), experiment with hybrid retrieval (dense + sparse), and implement re-ranking for precision.

**Related Patterns:** (1) Preloaded Context for static information; (3) Chain of Thought for multi-step reasoning over retrieved content; (6) Tool Use for accessing structured data sources.

---

### Pattern 3: Chain of Thought

**Problem:** Complex reasoning tasks—especially those involving multiple steps, numerical computation, or logical deduction—produce unreliable results when the LLM attempts to answer directly.

**Context:** The user's question requires the LLM to perform multi-step reasoning: mathematical problem-solving, causal analysis, planning, or any task where intermediate steps matter.

**Solution:** Prompt the LLM to externalize its reasoning process, generating intermediate steps before arriving at a final answer. This "thinking out loud" approach dramatically improves accuracy on complex tasks.

Implementation approaches:

- **Zero-shot CoT:** Simply append "Let's think step by step" to the prompt
- **Few-shot CoT:** Provide examples that demonstrate the reasoning format
- **Structured CoT:** Request specific reasoning steps (e.g., "First, identify the key facts. Second, determine the relationships...")

The mechanism appears to work because:
1. It forces the model to allocate tokens (computational resources) to reasoning
2. Intermediate steps provide self-correction opportunities
3. The explicit chain helps maintain coherence across complex problems

**Therefore:** For any task involving reasoning, planning, or multi-step analysis, explicitly request step-by-step thinking. Consider providing a reasoning template appropriate to the domain. Be aware that CoT increases token usage—it's a trade-off of cost for accuracy.

**Related Patterns:** (2) RAG combined with CoT (see "Retrieval-Augmented Thoughts"); (9) Reflection Loop for self-critique of reasoning; (15) Multi-Mind Analysis for diverse reasoning approaches.

---

### Pattern 4: System Prompt Architecture

**Problem:** As applications grow in complexity, system prompts become unwieldy, inconsistent, and difficult to maintain. Changes in one area break behavior in another.

**Context:** Your system prompt has grown beyond a few paragraphs. You have multiple contributors. You need to version, test, and iterate on prompt components independently.

**Solution:** Treat system prompts as structured documents with clear sections, explicit dependencies, and modular components. Apply software engineering principles: separation of concerns, single responsibility, explicit interfaces.

A recommended architecture:

```
1. IDENTITY BLOCK
   - Who is the assistant?
   - What organization/product does it represent?
   
2. CAPABILITIES BLOCK  
   - What tools are available?
   - What actions can it take?
   
3. KNOWLEDGE BLOCK
   - Domain-specific information
   - Reference materials
   
4. BEHAVIORAL BLOCK
   - Tone and style guidelines
   - Constraint and safety rules
   
5. FORMAT BLOCK
   - Output structure requirements
   - Citation/attribution rules
   
6. CONTEXTUAL BLOCK (dynamic)
   - User information
   - Session state
   - Temporal context
```

**Therefore:** Design system prompts as layered architectures. Use XML-style tags or clear headers to delimit sections. Keep sections independently testable. Document dependencies between sections. Version your prompts like code.

**Related Patterns:** (1) Preloaded Context as a key component; (14) Memory Layer for dynamic sections; (9) Tool Use for capability definition.

---

### Pattern 5: Prefilling (Putting Words in the Model's Mouth)

**Problem:** LLMs often begin responses with unnecessary preamble, inconsistent formatting, or may drift into an unintended response style before getting to the substance.

**Context:** You need precise control over how the LLM begins its response—perhaps forcing JSON output, maintaining a character persona, or skipping conversational pleasantries.

**Solution:** Start the assistant's response with pre-filled tokens that constrain and guide the generation. Most LLM APIs allow you to provide the beginning of the assistant message; the model then continues from that point.

Common prefilling techniques:

1. **Format forcing:** Prefill with `{` to force JSON output, or `<result>` to enforce XML structure
   
2. **Persona anchoring:** Prefill with `[CHARACTER_NAME]:` to keep the model in character during roleplay
   
3. **Preamble skipping:** Prefill with the first word of the desired content (e.g., `The answer is`) to bypass "Certainly!" or "Great question!"

4. **Language control:** Prefill with tokens in the target language to prevent the model from responding in the wrong language

5. **Structured extraction:** Prefill with `<name>` when extracting structured data to guide the model into the expected format

Implementation example:
```python
messages = [
    {"role": "user", "content": "Convert this to JSON: ..."},
    {"role": "assistant", "content": "{"}  # Prefilled
]
# Model continues from "{" and produces valid JSON
```

**Therefore:** Use prefilling as a lightweight steering mechanism. A few tokens of prefill can be more effective than paragraphs of instruction. Be mindful that prefilling works best for format control; it cannot override strong semantic tendencies.

**Related Patterns:** (4) System Prompt Architecture for instruction-based control; (23) Structured Output Enforcement for comprehensive format guarantees; (3) Chain of Thought can be triggered via prefilling ("Let me think through this...").

---

### Pattern 6: In-Context Transfer Learning

**Problem:** You need the LLM to perform a task for which you have examples, but fine-tuning is expensive, slow, or impractical. Zero-shot performance is inadequate.

**Context:** You have a specific task or domain where you possess examples of good input-output pairs, and you need better performance than prompting alone provides.

**Solution:** Include carefully selected examples (demonstrations) directly in the prompt. The LLM learns the task pattern from these examples without any weight updates—transferring knowledge purely through the context window.

The spectrum of in-context learning:

1. **Zero-shot:** No examples; rely on instructions alone
2. **One-shot:** Single example demonstrating the desired behavior
3. **Few-shot:** Multiple examples (typically 3-10) covering task variations
4. **Many-shot:** Large numbers of examples (dozens to hundreds) when context permits

Principles for effective demonstrations:

- **Diversity:** Examples should cover the range of expected inputs
- **Relevance:** Examples similar to the current query work best (consider dynamic retrieval)
- **Format consistency:** All examples should follow identical structure
- **Quality over quantity:** A few excellent examples beat many mediocre ones
- **Order matters:** Place the most relevant example last (nearest to the query)

Advanced variations:

- **Self-Generated ICL:** Have the LLM generate its own examples for the task
- **Retrieval-Augmented ICL:** Dynamically retrieve relevant examples for each query
- **Cross-task transfer:** Use examples from related tasks when target task data is scarce

**Therefore:** Build a library of high-quality examples for your key tasks. Implement dynamic example selection based on query similarity. Experiment with example count—more isn't always better due to attention dilution.

**Related Patterns:** (2) RAG for retrieving relevant examples; (3) Chain of Thought for reasoning demonstrations; (28) Prompt Self-Improvement for optimizing example selection.

---

### Pattern 7: Context Ablation

**Problem:** Prompts accumulate content over time—instructions, examples, retrieved documents—until they become bloated, expensive, and potentially counterproductive due to attention dilution.

**Context:** Your context window is filling up, costs are climbing, latency is increasing, or you suspect that too much context is actually hurting performance (the "lost in the middle" phenomenon).

**Solution:** Systematically remove or compress portions of the context, measuring the impact on output quality. Use this analysis to identify what context is actually necessary and what can be pruned.

Context ablation techniques:

1. **Ablation testing:** Remove sections one at a time and measure output quality changes. This reveals which context actually contributes.

2. **Token-level pruning:** Use a smaller model to calculate per-token importance (perplexity-based), removing low-importance tokens. Tools like LLMLingua achieve 20x compression with minimal quality loss.

3. **Chunk-level pruning:** For RAG systems, evaluate chunk relevance to the query and drop low-scoring chunks before feeding to the LLM.

4. **Progressive summarization:** Replace older conversation history with summaries, preserving essential information in fewer tokens.

5. **Query-aware compression:** Compress context differently based on the current query, preserving sections relevant to what's being asked.

What ablation studies typically reveal:

- Specific instructions matter more than examples (often)
- Recent context matters more than older context
- Middle positions receive less attention than beginning/end
- Much retrieved context is redundant or irrelevant

**Therefore:** Don't assume more context is better. Run ablation experiments to find your minimum viable context. Implement intelligent compression for long-running conversations or large document sets. Place critical information at the beginning or end of context, not the middle.

**Related Patterns:** (2) RAG for context retrieval that may need ablation; (1) Preloaded Context for optimizing static content; (27) Cost Control Layer for managing context-driven costs.

---

## Part II: Action and Tool Patterns

These patterns address the challenge of LLMs taking actions in the world—moving beyond text generation to actual task completion.

---

### Pattern 8: The Agent Loop

**Problem:** A single LLM call cannot complete complex, multi-step tasks that require observation, action, and iteration.

**Context:** You need the LLM to accomplish goals that require multiple steps: gathering information, taking actions, observing results, and adjusting approach—the kind of tasks that take humans minutes or hours, not seconds.

**Solution:** Implement an iterative loop where the LLM reasons about the current state, decides on an action, executes that action, observes the result, and repeats until the goal is achieved or a termination condition is met.

The canonical agent loop (often called "ReAct" for Reason + Act):

```
while not done:
    1. OBSERVE: Present current state to LLM
    2. THINK: LLM reasons about what to do next
    3. ACT: LLM selects an action/tool
    4. EXECUTE: System executes the action
    5. UPDATE: Incorporate results into state
    6. CHECK: Evaluate if goal is met or should terminate
```

Critical implementation considerations:

- **Termination conditions:** Max iterations, success criteria, error thresholds
- **State management:** What history to include, what to summarize or drop
- **Error handling:** How to recover from failed actions, invalid tool calls
- **Cost control:** Token limits, time limits, billing caps

**Therefore:** Build agent loops with robust termination conditions—infinite loops are a real risk. Implement comprehensive logging for debugging. Start with tight iteration limits and expand as you gain confidence. Always include human-in-the-loop escape hatches for high-stakes actions.

**Related Patterns:** (9) Tool Use as the action mechanism; (10) Sandbox and Proxy for safe execution; (12) Reflection Loop for self-correction; (13) Planning Pattern for complex goals.

---

### Pattern 9: Tool Use (Function Calling)

**Problem:** LLMs can only generate text, but real tasks require actions: querying databases, calling APIs, manipulating files, executing code.

**Context:** Your application needs the LLM to do more than converse—it needs to take concrete actions in systems, access live data, or perform computations beyond text manipulation.

**Solution:** Define a set of functions (tools) that the LLM can request to invoke. The LLM generates structured requests specifying which function to call and with what arguments; the system executes these functions and returns results.

Tool design principles:

1. **Clear descriptions:** The LLM only knows what you tell it. Tool descriptions are prompts.
2. **Atomic operations:** Each tool should do one thing well
3. **Predictable outputs:** Return structured, parseable results
4. **Error surfaces:** Return informative errors the LLM can reason about
5. **Idempotency where possible:** Safe to retry on failure

Modern implementations typically use:
- **OpenAI function calling format:** JSON schema for parameters
- **Model Context Protocol (MCP):** Anthropic's open standard for tool integration
- **Custom formats:** XML tags, special tokens, or structured prompts

**Therefore:** Design tools as a minimal, orthogonal set. Each tool should have a clear purpose, well-defined inputs, and predictable outputs. Document tools thoroughly—the documentation *is* the interface for the LLM. Test tools independently before giving them to agents.

**Related Patterns:** (8) Agent Loop as the orchestration layer; (10) Sandbox and Proxy for execution safety; (11) File Read/Write Tools as a fundamental tool category.

---

### Pattern 10: Sandbox and Proxy

**Problem:** Giving an LLM the ability to execute code or take actions creates security risks—both from adversarial prompts (prompt injection) and from honest mistakes (unintended consequences).

**Context:** Your application includes (6) Tool Use or code execution capabilities. The LLM will be processing untrusted input (user messages) and making decisions about actions to take.

**Solution:** Execute all LLM-initiated actions in isolated, constrained environments. Implement proxies that validate, log, and can block or transform requests before they reach real systems.

The defense-in-depth approach:

1. **Sandboxed execution:** Run code in containers with minimal permissions, read-only filesystems, no network access (unless explicitly needed)

2. **Capability boundaries:** Restrict available tools based on the task, user permissions, and trust level

3. **Proxy validation:** All tool calls pass through a validation layer that:
   - Checks parameters against schemas
   - Enforces rate limits
   - Validates against allowlists/blocklists
   - Logs for audit

4. **Output filtering:** Sanitize results before returning to users (prevent data exfiltration)

5. **Human approval gates:** For high-stakes actions, require explicit human confirmation

**Therefore:** Never trust LLM-generated code or tool calls implicitly. Design your system assuming the LLM might attempt anything—because prompt injection can make it do exactly that. Implement defense in depth: sandboxes, proxies, validation, logging, and human oversight for sensitive operations.

**Related Patterns:** (9) Tool Use as the capability being protected; (8) Agent Loop for implementing approval gates; (20) Privilege Boundaries for systematic access control.

---

### Pattern 11: File Read/Write Tools

**Problem:** Many useful tasks require reading from and writing to files—documents, code, data, configurations—but file system access is both powerful and dangerous.

**Context:** You're building an agent that needs to work with files: a coding assistant that edits source code, a document processor that transforms files, or an analyst that reads and writes reports.

**Solution:** Implement specialized tools for file operations that provide the capability while enforcing safety constraints.

A robust file tool set typically includes:

```
READ OPERATIONS:
- list_directory(path) → file listing
- read_file(path) → file contents
- search_files(pattern, path) → matching files
- view_range(path, start, end) → partial file contents

WRITE OPERATIONS:
- create_file(path, content) → creates new file
- str_replace(path, old, new) → surgical edits
- append_file(path, content) → adds to end

METADATA OPERATIONS:
- file_exists(path) → boolean
- file_info(path) → size, modified date, etc.
```

Key design decisions:

- **Prefer surgical edits over full rewrites:** `str_replace` is safer than overwriting entire files—it's harder to accidentally destroy content and easier to review changes

- **Enforce path constraints:** Jail all operations to designated directories. Never allow `../` traversal.

- **Create before overwriting:** Maintain backups or use copy-on-write for destructive operations

- **Size limits:** Cap file sizes for both reading (context window limits) and writing (storage limits)

**Therefore:** Build file tools that make the common case easy and the dangerous case hard. Prefer additive or surgical operations over destructive ones. Implement strict path validation. Provide good feedback about file state so the LLM can reason about its operations.

**Related Patterns:** (10) Sandbox and Proxy for constraining file access; (9) Tool Use as the general framework; (8) Agent Loop for iterating on file operations.

---

### Pattern 12: Reflection Loop

**Problem:** LLMs make mistakes—hallucinations, logical errors, incomplete analyses—but often can identify and correct these mistakes if prompted to review their work.

**Context:** You need higher reliability than a single LLM generation provides, especially for tasks where errors are costly or where the LLM's output will be used downstream without human review.

**Solution:** After generating an initial response, have the LLM (or a separate LLM instance) critique and revise its own output. This self-reflection cycle can iterate until quality criteria are met.

The basic reflection pattern:

```
1. GENERATE: Produce initial response
2. CRITIQUE: "Review the above response. What are its weaknesses, errors, or omissions?"
3. REVISE: "Given that critique, produce an improved response"
4. (Optional) ITERATE: Repeat 2-3 until satisfactory
```

Variations:

- **Evaluator-Optimizer:** Separate models for generation and evaluation
- **Constitutional AI:** Critique against explicit principles
- **Debate:** Multiple LLMs argue different positions
- **Verification:** Check specific claims against sources

**Therefore:** Build reflection loops for high-stakes outputs. The marginal cost of an extra LLM call is often far less than the cost of an error. Consider using different temperatures or prompts for generation vs. critique. Set iteration limits to prevent infinite loops.

**Related Patterns:** (3) Chain of Thought as input to reflection; (18) Multi-Mind Analysis for diverse critique perspectives; (8) Agent Loop for iterative improvement.

---

### Pattern 13: Planning Pattern

**Problem:** Complex goals require decomposition into sub-goals, but LLMs trying to solve everything in one step often miss dependencies, forget constraints, or pursue inefficient paths.

**Context:** The user's request is a high-level goal ("build me a website," "analyze this dataset and report findings") rather than a specific action. Success requires multiple coordinated steps.

**Solution:** Before acting, have the LLM generate an explicit plan—a sequence or graph of steps—then execute the plan systematically, re-planning as needed when circumstances change.

Planning approaches:

1. **Linear planning:** Sequence of steps executed in order
2. **DAG planning:** Steps with dependencies, parallelizable where independent
3. **Hierarchical planning:** High-level plan decomposed into sub-plans
4. **Adaptive planning:** Plan created incrementally as execution reveals information

A robust planning implementation:

```
1. ANALYZE: Understand the goal and constraints
2. DECOMPOSE: Break into sub-tasks
3. SEQUENCE: Order by dependencies
4. ESTIMATE: Assess effort/risk for each step
5. EXECUTE: Work through plan systematically
6. MONITOR: Check progress, re-plan if needed
```

**Therefore:** For complex goals, require explicit planning before action. Make plans visible to users for validation. Build in checkpoints for human review on long-running plans. Expect and handle plan revision—reality rarely matches initial assumptions.

**Related Patterns:** (8) Agent Loop for plan execution; (3) Chain of Thought for planning reasoning; (12) Reflection Loop for plan validation.

---

### Pattern 14: Skills (Modular Capabilities)

**Problem:** As LLM applications grow, they accumulate tools, prompts, and procedures that become difficult to maintain, reuse, or compose. Different tasks require different subsets of capabilities.

**Context:** You're building an agent that needs different capabilities for different situations—writing code vs. writing prose, analyzing data vs. generating images. You want these capabilities to be modular and reusable.

**Solution:** Package related capabilities into "skills"—modular bundles that include tools, specialized prompts, examples, and behavioral guidelines. Skills can be loaded dynamically based on the task at hand.

A skill typically includes:

1. **Tools:** Functions the LLM can call when this skill is active
2. **Instructions:** Specialized prompts for how to use the skill effectively  
3. **Examples:** Demonstrations of the skill in use
4. **Guidelines:** Best practices, common pitfalls, output formats

Example skill structure:
```
skills/
├── code_review/
│   ├── SKILL.md          # Instructions and guidelines
│   ├── tools/            # Static analysis, linting tools
│   └── examples/         # Good/bad code review examples
├── data_analysis/
│   ├── SKILL.md
│   ├── tools/            # Pandas, plotting functions
│   └── examples/
└── document_writing/
    ├── SKILL.md
    ├── tools/            # Formatting, citation tools
    └── examples/
```

Skill activation strategies:

- **Static:** Load skills based on application mode or user selection
- **Dynamic:** Let the LLM choose which skills to activate for a task
- **Hierarchical:** Meta-skills that compose other skills

**Therefore:** Design capabilities as self-contained skills that can be loaded, combined, and versioned independently. Include both the tools *and* the knowledge of how to use them effectively. Build a skill discovery mechanism so agents can find and apply relevant skills.

**Related Patterns:** (9) Tool Use as skill components; (4) System Prompt Architecture for skill integration; (17) Router Pattern for skill selection.

---

### Pattern 15: Test-Based Grounding

**Problem:** LLMs generate code or structured outputs that look plausible but may contain subtle errors. Text-based review catches some issues but misses runtime failures.

**Context:** The LLM is producing code, queries, configurations, or other artifacts that can be executed or validated programmatically. You need confidence that the output actually works.

**Solution:** Validate LLM outputs by executing tests against them. Feed test results back to the LLM as concrete grounding for iteration. Treat test execution as a form of reality checking.

Implementation approaches:

1. **Write-and-test loop:** LLM writes code, tests execute, failures feed back for revision
   ```
   while not tests_pass:
       code = llm.generate(prompt + error_feedback)
       results = run_tests(code)
       if results.failed:
           error_feedback = format_failures(results)
   ```

2. **Test-first generation:** Provide tests upfront, have LLM write code to satisfy them (TDD-style)

3. **Property-based verification:** Generate test cases that check invariants rather than specific examples

4. **Execution-guided feedback:** Run the code, observe runtime behavior, report back to LLM

Benefits of test-based grounding:

- **Objectivity:** Tests provide unambiguous pass/fail signals
- **Specificity:** Error messages point to exact failures
- **Iteration:** Clear feedback enables targeted fixes
- **Confidence:** Passing tests provide evidence of correctness

**Therefore:** When generating executable artifacts, integrate test execution into your pipeline. Prefer fast, isolated tests that provide clear feedback. Design error messages to be informative for the LLM. Consider both unit tests (specific behaviors) and integration tests (overall functionality).

**Related Patterns:** (12) Reflection Loop where tests serve as the critic; (8) Agent Loop for iterative test-fix cycles; (10) Sandbox and Proxy for safe test execution.

---

## Part III: Multi-Model and Orchestration Patterns

These patterns address architectures involving multiple LLMs or complex coordination.

---

### Pattern 16: Memory Layer

**Problem:** LLM conversations are stateless—each interaction starts fresh. But users expect continuity: remembering preferences, past conversations, established facts, and ongoing projects.

**Context:** Your application has users who return over multiple sessions. They expect the system to remember relevant information without re-explaining everything each time.

**Solution:** Implement a persistent memory layer that stores, retrieves, and manages information across sessions. This layer sits between the user and the LLM, enriching prompts with relevant history.

Memory architecture typically includes:

1. **Short-term memory:** Recent conversation history (last N messages or summarized)
2. **Long-term memory:** Persistent facts about the user (preferences, background, relationships)
3. **Working memory:** Current task context (active project, in-progress analysis)
4. **Episodic memory:** Summaries of past sessions, searchable by topic or time

Implementation considerations:

- **What to store:** Extracting memory-worthy facts from conversations
- **When to retrieve:** Determining what past context is relevant to current query
- **How to present:** Formatting memories for LLM consumption
- **When to forget:** Handling outdated, contradictory, or user-deleted information

**Therefore:** Design memory as a first-class system, not an afterthought. Decide what information deserves persistence. Build retrieval that balances relevance and recency. Give users visibility into and control over what is remembered.

**Related Patterns:** (2) RAG for memory retrieval mechanics; (1) Preloaded Context for incorporating memories; (4) System Prompt Architecture for memory presentation.

---

### Pattern 17: Router Pattern

**Problem:** Different queries require different handling—some need tools, some need specific models, some need particular prompts—but the user shouldn't have to specify this.

**Context:** Your application serves diverse request types. A one-size-fits-all approach either over-provisions (expensive) or under-provisions (low quality) for individual requests.

**Solution:** Implement a routing layer that classifies incoming requests and directs them to appropriate handlers—different models, prompts, tool sets, or processing pipelines.

Router architectures:

1. **Classifier router:** LLM or traditional ML model classifies intent, routes to handler
2. **Capability router:** Routes based on required capabilities (code, search, reasoning)
3. **Cost router:** Routes based on complexity—simple queries to cheap models, hard queries to expensive ones
4. **Semantic router:** Embeds queries and routes by similarity to exemplars

The router itself can be:
- A small, fast LLM optimized for classification
- A traditional ML classifier
- A rule-based system with heuristics
- An embedding-similarity lookup

**Therefore:** Build routing when your application spans multiple domains, capabilities, or cost tiers. Keep the router fast and cheap—it runs on every request. Monitor routing accuracy and adjust based on downstream outcomes.

**Related Patterns:** (18) Multi-Mind Analysis for routing to multiple analysts; (9) Tool Use for routing based on required tools; (4) System Prompt Architecture for handler-specific prompts.

---

### Pattern 18: Deterministic Chain

**Problem:** Some workflows have a fixed, known structure—the same steps every time. Using an agent loop for these adds unnecessary latency and unpredictability.

**Context:** You have a well-defined process (e.g., "extract info → validate → transform → store") where the steps don't vary based on intermediate results.

**Solution:** Implement workflows as deterministic chains of LLM calls, where each step's output feeds the next step's input in a fixed sequence.

Chain design:

```
Step 1: EXTRACT
  Input: Raw document
  Output: Structured data
  
Step 2: VALIDATE  
  Input: Structured data
  Output: Validated data + error flags
  
Step 3: TRANSFORM
  Input: Validated data  
  Output: Target format

Step 4: COMMIT
  Input: Target format
  Output: Confirmation
```

Advantages over agent loops:
- Predictable execution time
- Easier debugging (each step isolated)
- No risk of infinite loops
- Can be partially automated (some steps could be non-LLM)

**Therefore:** Use chains for well-understood processes. Reserve agent loops for genuinely open-ended tasks. Chains can include conditional branches, but the branch structure itself should be predetermined.

**Related Patterns:** (8) Agent Loop for open-ended tasks; (13) Planning Pattern for determining when chains are appropriate; (9) Tool Use within chain steps.

---

### Pattern 19: Dual LLM Pattern

**Problem:** A single LLM processing untrusted input (user messages, retrieved documents) risks prompt injection—where malicious content in the input manipulates the LLM's behavior.

**Context:** Your application retrieves external content, processes user-uploaded documents, or handles any input that could contain adversarial prompts.

**Solution:** Separate the architecture into two distinct LLM instances with different trust levels:

1. **Privileged LLM:** Receives system instructions and plans actions. Never sees raw untrusted content.

2. **Quarantined LLM:** Processes untrusted content. Has no access to tools or sensitive operations. Can only return sanitized, structured outputs.

Communication between them flows through a traditional software layer (the "orchestrator") that enforces constraints:

```
User Input → [Quarantined LLM] → Sanitized Summary → 
[Orchestrator validates] → [Privileged LLM] → Actions
```

The quarantined LLM might:
- Summarize documents into structured data
- Extract specific fields with constrained formats
- Answer specific questions about content

The privileged LLM then operates on these sanitized outputs without ever seeing the original untrusted content.

**Therefore:** For high-security applications, implement strict separation between LLMs that see untrusted content and LLMs that can take actions. The boundary between them should be enforced by traditional software, not by prompts.

**Related Patterns:** (10) Sandbox and Proxy for execution isolation; (20) Privilege Boundaries for systematic access control; (9) Tool Use restricted to privileged context.

---

### Pattern 20: Multi-Mind Analysis

**Problem:** A single LLM's analysis reflects its particular biases, blind spots, and reasoning patterns. Important decisions benefit from diverse perspectives.

**Context:** You're building a system for analysis, decision support, or creative generation where diversity of thought adds value—investment analysis, risk assessment, research synthesis, creative brainstorming.

**Solution:** Invoke multiple LLM "minds"—either different models, different prompts, or different personas—to analyze the same input from different perspectives, then synthesize their outputs.

Implementation variations:

1. **Same model, different prompts:** "Analyze as an optimist," "Analyze as a skeptic," "Analyze as a domain expert"

2. **Different models:** Claude for nuance, GPT for breadth, Gemini for multimodal

3. **Different temperatures:** Low temperature for conservative analysis, high for creative alternatives

4. **Debate format:** Models critique each other's positions

5. **Ensemble voting:** Multiple models vote on discrete decisions

Synthesis approaches:
- **Aggregation:** Combine all perspectives into comprehensive analysis
- **Arbitration:** Third LLM evaluates and reconciles differences
- **Transparency:** Present all perspectives with disagreements highlighted

**Therefore:** For high-stakes analysis, generate multiple perspectives. Make diversity explicit—different prompts, models, or configurations. Present synthesis that preserves dissent rather than papering over disagreements.

**Related Patterns:** (12) Reflection Loop as a two-perspective case; (17) Router Pattern for directing to specialists; (3) Chain of Thought for each perspective's reasoning.

---

## Part IV: Safety and Control Patterns

These patterns address the critical challenge of building LLM applications that remain safe, controllable, and aligned with human intent.

---

### Pattern 21: Action Selector Pattern

**Problem:** LLM agents with full autonomy over tool selection can be manipulated by prompt injection to invoke unintended tools or misuse intended tools.

**Context:** Your agent has access to multiple tools, including some with significant side effects. You need to constrain tool use to expected patterns.

**Solution:** The LLM acts only as a *selector* among pre-defined actions, not as a general-purpose action generator. The LLM's only output is which pre-defined action to invoke. It never sees tool execution results, preventing feedback loops that could be exploited.

```
User: "Send this email to my team"

LLM (action selector) → ACTION: send_email
[Traditional code composes email from user input]
[No LLM-generated content in email body]
```

This pattern offers maximum security but limited flexibility. The LLM is reduced to a natural language interface for a fixed set of operations.

**Therefore:** For highest-security applications, constrain the LLM to action selection only. No generated content reaches external systems. The trade-off is flexibility for security.

**Related Patterns:** (9) Tool Use as the general case; (19) Dual LLM Pattern for separation of concerns; (10) Sandbox and Proxy for defense in depth.

---

### Pattern 22: Privilege Boundaries

**Problem:** As LLM applications grow, different operations require different permission levels, but permissions are often implemented ad-hoc, leading to security gaps.

**Context:** Your application has operations ranging from low-risk (reading public data) to high-risk (financial transactions, data deletion, access credential management).

**Solution:** Design explicit privilege boundaries with formally defined tiers. Operations at each tier require appropriate authentication, authorization, and confirmation.

A typical privilege tier structure:

```
TIER 0: READ-ONLY
- Query public information
- Generate text responses
- Read from designated safe sources

TIER 1: LIMITED WRITE
- Create draft documents
- Modify user's own data
- Send notifications (with confirmation)

TIER 2: SIGNIFICANT ACTION  
- Send communications on user's behalf
- Modify shared resources
- Execute code in sandboxes

TIER 3: SENSITIVE OPERATIONS
- Access credentials or secrets
- Modify access permissions
- Financial transactions
Requires: Explicit human approval each time

TIER 4: ADMINISTRATIVE
- System configuration changes
- User management
- Audit log access
Requires: Out-of-band authentication
```

**Therefore:** Map every tool and action to a privilege tier. Implement tier-appropriate controls. Assume any LLM-adjacent code might be manipulated—privilege enforcement must be in traditional code, not prompts.

**Related Patterns:** (10) Sandbox and Proxy for tier-appropriate isolation; (21) Action Selector for reducing attack surface; (19) Dual LLM Pattern for privilege separation.

---

### Pattern 23: Structured Output Enforcement

**Problem:** LLMs generate freeform text, but downstream systems need structured data. Malformed outputs cause errors; manipulated outputs cause security issues.

**Context:** Your application parses LLM output for use in APIs, databases, or rendered interfaces. Reliability and security depend on predictable structure.

**Solution:** Enforce structured output at multiple levels: prompting, parsing, and validation.

Enforcement layers:

1. **Prompting:** Request specific formats (JSON, XML) with schemas

2. **Constrained generation:** Use API features that restrict output to valid JSON (available in recent models)

3. **Parsing with fallbacks:** Attempt to parse, request correction on failure

4. **Schema validation:** Validate parsed output against strict schemas

5. **Semantic validation:** Check that values make sense (dates in valid ranges, enums in valid sets)

```
Generated → Parse → Schema Validate → Semantic Validate → Use
              ↓           ↓                    ↓
           Retry      Retry/Fail          Retry/Fail
```

**Therefore:** Never trust that LLM output will match requested format. Implement defense in depth: request structure, constrain generation if possible, validate thoroughly, and handle malformed output gracefully.

**Related Patterns:** (9) Tool Use for structured tool calls; (10) Sandbox and Proxy for output sanitization; (18) Deterministic Chain for predictable outputs.

---

### Pattern 24: Human-in-the-Loop Gates

**Problem:** Fully autonomous LLM systems can make costly mistakes. Users need oversight and control, especially for irreversible or high-stakes actions.

**Context:** Your application can take actions with real-world consequences—sending communications, modifying data, executing transactions, or deploying code.

**Solution:** Implement explicit gates where human confirmation is required before proceeding. Make the pending action clearly visible and easily cancellable.

Gate design principles:

1. **Clear action preview:** Show exactly what will happen
2. **Easy approval:** One-click confirmation for routine actions
3. **Easy rejection:** One-click cancel, option to modify
4. **Timeout handling:** Default to safe state if no response
5. **Audit trail:** Log all approvals and rejections

Gate triggers:
- **Always:** Certain action types always require approval
- **Threshold:** Actions above certain impact levels
- **Uncertainty:** When the LLM reports low confidence
- **Anomaly:** Actions outside typical patterns

**Therefore:** Design human-in-the-loop as a feature, not an obstacle. Make approval frictionless for routine actions while ensuring users genuinely review high-stakes ones. Build escape hatches for fully stopping automated processes.

**Related Patterns:** (8) Agent Loop for implementing approval checkpoints; (22) Privilege Boundaries for gate placement; (13) Planning Pattern for previewing planned actions.

---

## Part V: Infrastructure Patterns

These patterns address the operational infrastructure supporting LLM applications.

---

### Pattern 25: Model Context Protocol (MCP)

**Problem:** Every combination of LLM application and external tool requires custom integration work. This "M×N" problem makes tool ecosystems fragmented and expensive to maintain.

**Context:** You're building or integrating tools for LLM applications and want them to work across multiple LLM providers and client applications.

**Solution:** Adopt the Model Context Protocol—an open standard that defines how LLM applications connect to external data sources and tools. MCP provides a universal interface based on JSON-RPC 2.0.

MCP defines three core primitives:

1. **Resources:** Structured data sources the LLM can read (files, database rows, API responses)

2. **Tools:** Functions the LLM can invoke (CRUD operations, API calls, computations)

3. **Prompts:** Reusable prompt templates that servers can expose

Architecture:
- **MCP Hosts:** LLM applications (Claude Desktop, IDE plugins, custom apps) with embedded MCP clients
- **MCP Servers:** Services exposing resources, tools, and prompts
- **Transport:** JSON-RPC over stdio, HTTP, or websockets

**Therefore:** When building LLM tools, implement them as MCP servers for maximum reusability. When building LLM applications, implement MCP client support to access the growing ecosystem of MCP servers.

**Related Patterns:** (9) Tool Use for what MCP standardizes; (10) Sandbox and Proxy for MCP security considerations.

---

### Pattern 26: Observability Layer

**Problem:** LLM applications are difficult to debug—the "reasoning" happens inside opaque model calls, failures may be semantic rather than technical, and problems may only emerge over many interactions.

**Context:** You're operating an LLM application in production and need to understand its behavior, diagnose issues, and improve over time.

**Solution:** Implement comprehensive observability covering all LLM interactions:

1. **Logging:**
   - Full prompts and responses (or hashes for privacy)
   - Token counts, latencies, costs
   - Tool calls and results
   - Error conditions and retries

2. **Tracing:**
   - Request IDs through entire pipelines
   - Parent-child relationships in agent loops
   - Cross-service correlation

3. **Metrics:**
   - Response quality scores (user feedback, automated eval)
   - Latency distributions
   - Error rates by type
   - Cost per request, per user, per task type

4. **Evaluation:**
   - Regular evaluation against test sets
   - Regression detection on prompt changes
   - A/B testing infrastructure for improvements

**Therefore:** Instrument everything. LLM applications fail in subtle ways—you need data to diagnose issues. Build evaluation into your development process, not just production monitoring.

**Related Patterns:** (8) Agent Loop for tracing complex flows; (12) Reflection Loop for automated evaluation; (4) System Prompt Architecture for versioning tracked changes.

---

### Pattern 27: Cost Control Layer

**Problem:** LLM API costs can spiral unexpectedly—long conversations, runaway agent loops, or inefficient prompts can generate surprising bills.

**Context:** You're operating at scale or allowing users to trigger LLM calls, creating cost exposure that needs management.

**Solution:** Implement explicit cost controls at multiple levels:

1. **Per-request limits:** Maximum tokens per completion
2. **Per-conversation limits:** Maximum total tokens before requiring new session
3. **Per-user limits:** Daily/monthly quotas
4. **Per-loop limits:** Maximum iterations in agent loops
5. **Global limits:** Circuit breakers for total spend

Implementation:
- Track token usage in real-time
- Implement budget enforcement before requests, not just after
- Build graceful degradation (summarize context rather than fail)
- Alert on unusual patterns before limits are hit

**Therefore:** Treat cost as a first-class system property. Budget, track, and enforce. Build alerts that fire before disaster. Implement graceful degradation over hard failures.

**Related Patterns:** (8) Agent Loop limits for iteration control; (17) Router Pattern for cost-aware model selection; (26) Observability Layer for cost tracking.

---

## Part VI: Optimization and Evaluation Patterns

These patterns address improving LLM application quality through systematic optimization and evaluation.

---

### Pattern 28: Prompt Self-Improvement (DSPy-style Optimization)

**Problem:** Manual prompt engineering is time-consuming, doesn't transfer well between models, and often leaves significant performance on the table. Humans struggle to explore the space of possible prompts systematically.

**Context:** You have a task with measurable quality metrics and a set of representative examples. You want to optimize prompts programmatically rather than through trial and error.

**Solution:** Use a framework like DSPy that treats prompts as optimizable parameters. Define your task declaratively (inputs and outputs), provide training data, and let the optimizer discover effective prompts automatically.

DSPy's approach:

1. **Signatures:** Declare what you want (e.g., `question -> answer`) rather than how to get it
2. **Modules:** Compose signatures into pipelines (ChainOfThought, ReAct, etc.)
3. **Optimizers:** Automatically improve prompts based on training data and metrics
   - **BootstrapFewShot:** Selects optimal few-shot examples
   - **MIPROv2:** Optimizes both instructions and examples
   - **COPRO:** Iteratively refines instructions via coordinate ascent

Key insight: The optimizer explores prompt variations systematically, finding combinations humans wouldn't try.

```python
# Define what, not how
class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=5)
        self.generate = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = self.retrieve(question)
        return self.generate(context=context, question=question)

# Optimize automatically
optimizer = dspy.MIPROv2(metric=answer_correctness)
optimized_rag = optimizer.compile(RAG(), trainset=training_data)
```

**Therefore:** For tasks with clear metrics, consider programmatic prompt optimization over manual engineering. This is especially valuable when switching models—the optimizer can re-tune for the new model automatically.

**Related Patterns:** (6) In-Context Transfer Learning for what optimizers often produce; (12) Reflection Loop for optimization-like iteration; (26) Observability Layer for tracking optimization experiments.

---

### Pattern 29: Rollout Evaluation (Multi-Sample Assessment)

**Problem:** LLM outputs are non-deterministic. A single generation might be unrepresentatively good or bad. Evaluating reliability requires understanding the distribution of possible outputs.

**Context:** You need confidence in LLM behavior for high-stakes applications, or you want to assess uncertainty in LLM outputs. A single test run provides insufficient information.

**Solution:** Generate multiple completions (rollouts) for the same input and evaluate the distribution. This reveals both the typical behavior and the variance.

Implementation approaches:

1. **Parallel sampling:** Generate N completions simultaneously (efficient)
2. **Temperature variation:** Sample at different temperatures to explore the output space
3. **Pass@k evaluation:** For code generation, measure how often at least one of k attempts succeeds

What rollout evaluation reveals:

- **Reliability:** How often does the model get it right?
- **Consistency:** Do different runs produce similar answers?
- **Uncertainty:** High variance suggests the model is uncertain
- **Failure modes:** What kinds of errors occur and how often?

Use cases:

- **Code generation:** Run tests on multiple completions, report pass@k
- **Factual QA:** Check if answers cluster around the truth
- **Uncertainty quantification:** Use output variance as confidence signal
- **Evaluation pipelines:** Average metrics across multiple samples for stable scores

```python
def rollout_evaluate(prompt, n_samples=5, temperature=0.8):
    completions = [generate(prompt, temp=temperature) for _ in range(n_samples)]
    scores = [evaluate(c) for c in completions]
    return {
        "mean_score": mean(scores),
        "std_score": std(scores),
        "pass_at_1": scores[0] > threshold,
        "pass_at_k": any(s > threshold for s in scores)
    }
```

**Therefore:** For serious evaluation, sample multiple outputs and report distributions rather than single-point estimates. Use rollout evaluation to identify unreliable behaviors before deployment. Consider higher temperatures to stress-test robustness.

**Related Patterns:** (20) Multi-Mind Analysis for diverse perspectives; (26) Observability Layer for tracking rollout metrics; (15) Test-Based Grounding for evaluating code completions.

---

## Conclusion: Using the Pattern Language

Like Alexander's original, this pattern language is not prescriptive but generative. No application will use all patterns; most will combine a subset appropriate to their domain.

A typical application might combine:

**Simple chatbot:**
- (1) Preloaded Context
- (4) System Prompt Architecture
- (23) Structured Output Enforcement

**RAG-based assistant:**
- (1) Preloaded Context
- (2) Retrieval-Augmented Generation
- (3) Chain of Thought
- (4) System Prompt Architecture
- (7) Context Ablation for optimizing retrieval

**Autonomous agent:**
- (1) Preloaded Context
- (8) Agent Loop
- (9) Tool Use
- (10) Sandbox and Proxy
- (11) File Read/Write Tools
- (12) Reflection Loop
- (14) Skills
- (15) Test-Based Grounding
- (22) Privilege Boundaries
- (24) Human-in-the-Loop Gates

**Multi-model analysis system:**
- (17) Router Pattern
- (20) Multi-Mind Analysis
- (12) Reflection Loop
- (19) Dual LLM Pattern

**Optimized production system:**
- (28) Prompt Self-Improvement
- (29) Rollout Evaluation
- (26) Observability Layer
- (27) Cost Control Layer

The patterns reference each other because they solve related problems. Start with the problem you face, find the relevant pattern, follow its references to discover related patterns you may need.

---

## Invitation to Contribute

This is a proposal and a beginning, not a definitive catalog. The field evolves rapidly, and practitioners are discovering new patterns while refining existing ones.

We invite contributions:

- **New patterns** for recurring problems not yet documented
- **Refinements** to existing patterns based on implementation experience
- **Counter-examples** where patterns fail or prove inappropriate
- **Connections** between patterns not yet identified
- **Names** that better capture pattern essence

The goal is a living document that serves practitioners—a shared vocabulary for the design decisions we all face when building LLM applications.

---

*This document was created in response to [@antiali.as](https://bsky.app/profile/antiali.as)'s observation on BlueSky that "Someone's gotta start compiling a Pattern Language for LLM applications." Consider it a first draft of that compilation.*

---

## References

1. Alexander, C., Ishikawa, S., & Silverstein, M. (1977). *A Pattern Language: Towns, Buildings, Construction*. Oxford University Press.

2. Anthropic. (2024). "Model Context Protocol." https://modelcontextprotocol.io

3. Anthropic. (2024). "Prefill Claude's response for greater output control." https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/prefill-claudes-response

4. Databricks. (2024). "Agent System Design Patterns." https://docs.databricks.com

5. Databricks. (2024). "Optimizing Databricks LLM Pipelines with DSPy." https://www.databricks.com/blog/optimizing-databricks-llm-pipelines-dspy

6. Debenedetti et al. (2024). "Design Patterns for Securing LLM Agents against Prompt Injections." arXiv.

7. Jiang, H., et al. (2023). "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models." EMNLP.

8. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS.

9. Microsoft Research. (2024). "LLMLingua: Prompt Compression." https://github.com/microsoft/LLMLingua

10. Khattab, O., et al. (2023). "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." arXiv.

11. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS.

12. Yao, S., et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR.

13. Agrawal, A., et al. (2024). "Many-Shot In-Context Learning." NeurIPS.

14. Various practitioners and open-source communities contributing to the emerging practices documented herein.
