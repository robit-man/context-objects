# Context Objects Experiment
Highly dynamic context object and observability framework for agent experimentation

Below is comprehensive API documentation for context.py, covering every public class, function, and method in turn. Each section describes purpose, parameters, return values, exceptions, thread-safety, and provides usage notes or examples where helpful.

⸻

Utility

default_clock() -> datetime

Return the current UTC time.
	•	Returns
datetime – the current UTC timestamp (equivalent to datetime.utcnow()).

now = default_clock()


⸻

MemoryTrace

@dataclass
class MemoryTrace:
    trace_id: str
    stage_id: Optional[str]
    timestamp: str
    coactivated_with: List[str]
    metadata: Dict[str, Any]

Description
Captures a single recall event for a ContextObject, recording when it was recalled, in which processing stage, and with which other contexts it co-activated.
	•	Fields
	•	trace_id (str): Unique UUID for this recall event.
	•	stage_id (Optional[str]): Identifier of the pipeline stage that triggered the recall.
	•	timestamp (str): UTC timestamp of the recall, formatted YYYYMMDDTHHMMSSZ.
	•	coactivated_with (List[str]): IDs of other contexts recalled simultaneously.
	•	metadata (Dict[str,Any]): Arbitrary data about the event (e.g. retrieval score).
	•	Thread-safety
Immutable except for its fields; safe to read from multiple threads.

⸻

ContextObject

@dataclass
class ContextObject:
    domain: str
    component: str
    semantic_label: str
    schema_version: int = field(init=False, default=1)
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: Optional[str] = None
    timestamp: str = field(default_factory=lambda: default_clock().strftime("%Y%m%dT%H%M%SZ"))
    segment_ids: List[str] = field(default_factory=list)
    stage_id: Optional[str] = None
    references: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    retrieval_score: Optional[float] = None
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    graph_node_id: Optional[str] = None
    memory_tier: Optional[str] = None
    memory_traces: List[MemoryTrace] = field(default_factory=list)
    association_strengths: Dict[str, float] = field(default_factory=dict)
    recall_stats: Dict[str, Any] = field(default_factory=lambda: {"count": 0, "last_recalled": None})
    firing_rate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    pinned: bool = False
    last_accessed: Optional[str] = None
    expires_at: Optional[str] = None
    acl: Dict[str, Any] = field(default_factory=dict)
    batch_id: Optional[str] = None
    dirty: bool = True

Description
The core unit of contextual memory. Holds arbitrary “segments,” “stages,” or “artifacts” for use in agent pipelines, self-improvement, or retrieval.

Initialization and Lifecycle
	•	__post_init__()
	•	Validates that domain, component, and semantic_label are provided.
	•	Normalizes domain into one of {"segment","stage","artifact"} (defaulting to "artifact").
	•	Ensures timestamp is a string in YYYYMMDDTHHMMSSZ format.
	•	Sets last_accessed to timestamp if not provided.
	•	Maps memory_tier by domain:
	•	"segment" → "STM"
	•	"stage"   → "LTM"
	•	"artifact"→ "WM"
	•	Thread-safety:
	•	An internal threading.Lock (self._lock) protects updates in record_recall.

Pickling Support
	•	__getstate__() / __setstate__()
Cleanly exclude and restore non-picklable internals (_lock, _logger).

Representation
	•	__repr__()
Returns "<ContextObject {domain}/{component}/{semantic_label}@{timestamp}>".

⸻

Core Helper Methods

touch() -> None

Mark this object as accessed right now.
	•	Updates last_accessed to current UTC time.
	•	Sets dirty = True.

ctx.touch()

set_expiration(ttl_seconds: int) -> None

Schedule expiration after ttl_seconds.
	•	Computes expires_at = now + ttl_seconds.
	•	Sets dirty = True.

ctx.set_expiration(3600)  # expire in 1 hour

compute_embedding(default_embedder: Callable[[str],List[float]], component_embedder: Optional[Dict[str,Callable[[str],List[float]]]] = None) -> None

Generate or update self.embedding from self.summary.
	•	Chooses from component_embedder[self.component] if available, otherwise default_embedder.
	•	No-op if summary is None.
	•	Sets dirty = True.

ctx.summary = "This is a test."
ctx.compute_embedding(my_embed, {"prompt": prompt_embed})

log_context(level: int = logging.INFO) -> None

Log the entire context as JSON at the given log level.

ctx.log_context(logging.DEBUG)

record_recall(stage_id: Optional[str], coactivated_with: Optional[List[str]] = None, retrieval_score: Optional[float] = None) -> None

Thread-safely register a recall event:
	1.	Appends a new MemoryTrace(stage_id, coactivated_with, metadata={"retrieval_score":…}).
	2.	Increments recall_stats["count"] and updates "last_recalled".
	3.	Recalculates firing_rate = 1/count.
	4.	Increments association_strengths for each coactivated_with ID.
	5.	Calls touch().

ctx.record_recall(stage_id="plan", coactivated_with=["id1","id2"], retrieval_score=0.85)


⸻

Serialization

to_dict() -> Dict[str,Any]

Return a JSON-serializable dict of all fields, including nested memory_traces.

to_json() -> str

Return json.dumps(self.to_dict()).

@classmethod from_dict(data: Dict[str,Any]) -> ContextObject

Reconstruct a ContextObject (and its MemoryTraces) from a dict produced by to_dict.

@staticmethod from_json(s: str) -> ContextObject

Parse JSON string into a ContextObject.

⸻

Factory Constructors

Each returns a new, appropriately-initialized ContextObject (dirty by default).
	•	make_narrative(entry: str, tags: Optional[List[str]] = None, **metadata) -> ContextObject
Deduplicates identical latest narrative; otherwise creates a new component="narrative" artifact.
	•	make_success(description: str, refs: List[str] = None) -> ContextObject
A success marker in the "stage" domain.
	•	make_failure(description: str, refs: List[str] = None) -> ContextObject
A failure marker in the "stage" domain.
	•	make_segment(semantic_label: str, content_refs: List[str], tags: Optional[List[str]] = None, **metadata) -> ContextObject
Domain "segment", component="segment".
	•	make_stage(stage_name: str, input_refs: List[str], output: Any, **metadata) -> ContextObject
Domain "stage", component/label = stage_name; stores output in metadata.
	•	make_tool_code(label: str, code: str, tags: Optional[List[str]] = None, **metadata) -> ContextObject
Artifact for code snippets; full code in metadata["code"].
	•	make_schema(label: str, schema_def: str, ...)
	•	make_prompt(label: str, prompt_text: str, ...)
	•	make_policy(label: str, policy_text: str, ...)
	•	make_knowledge(label: str, content: str, ...)

Each of the last four sets summary to the first 120 chars (with “…”), stores full text in metadata, and tags accordingly.

⸻

ContextRepository

class ContextRepository:
    def __init__(self, path: str)
    def get(self, context_id: str) -> ContextObject
    def save(self, ctx: ContextObject) -> None
    def exists(self, context_id: str) -> bool
    @classmethod instance() -> ContextRepository
    def delete(self, context_id: str) -> None
    def query(self, filter_func: Callable[[ContextObject],bool]) -> List[ContextObject]

Description
A simple JSONL-backed Data Access Object for persisting and retrieving ContextObjects.
	•	__init__(path)
Creates or touches the file at path, registers itself as the singleton.
	•	get(context_id)
Scans file line-by-line, returning the first matching ContextObject.
	•	Raises: KeyError if not found.
	•	save(ctx)
Appends ctx.to_json() to the file only if ctx.dirty is True, then sets dirty=False.
Thread-safe via an internal lock.
	•	exists(context_id) -> bool
Returns True if get() succeeds; otherwise False.
	•	instance() -> ContextRepository
Returns the singleton created in __init__;
	•	Raises RuntimeError if none exists yet.
	•	delete(context_id)
Removes all records with that ID by rewriting the file (lock-protected).
	•	query(filter_func)
Streams all records, applies filter_func (taking a ContextObject), and returns those where it returns True.

⸻

MemoryManager

class MemoryManager:
    def __init__(self, repo: ContextRepository)
    def recall(self, seed_ids: List[str], k: int = 5, weights: Optional[Dict[str,float]] = None) -> List[ContextObject]
    def decay_and_promote(self, half_life: float = 86400.0, promote_threshold: int = 3) -> None
    def reinforce(self, context_id: str, coactivated: List[str]) -> None
    def prune(self, ttl_hours: int) -> None

Description
High-level associative memory operations for recall, reinforcement, decay, promotion, and pruning.
	•	recall(seed_ids, k=5, weights=None) -> List[ContextObject]
	1.	For each seed ID, loads its association_strengths.
	2.	For each associated ID, computes:

score = strength*weights["assoc"] 
      + weights["recency"]/(1 + age_seconds)


	3.	Aggregates scores across seeds, sorts descending, returns top k contexts.
	•	Parameters
	•	seed_ids: list of context IDs to seed recall
	•	k: number of results
	•	weights: e.g. {"assoc":1.0,"recency":1.0}

	•	decay_and_promote(half_life=86400.0, promote_threshold=3) -> None
Periodically called to:
	1.	Decay each association_strength by exp(-delta/half_life).
	2.	Drop strengths below 1e-6.
	3.	If recall_stats["count"] >= promote_threshold, upgrades memory_tier→"LTM".
	4.	Saves any changed contexts.
	•	reinforce(context_id, coactivated) -> None
Safely strengthen associations for context_id with each in coactivated:
	•	Drops any dangling IDs (deleted contexts).
	•	Calls record_recall on the target, then saves.
	•	prune(ttl_hours) -> None
Deletes all un-pinned contexts whose last_accessed is older than ttl_hours ago.

⸻

ContextGraph

class ContextGraph:
    def __init__(self)
    def add_node(self, ctx: ContextObject) -> None
    def add_edge(self, from_id: str, to_id: str, label: Optional[str] = None) -> None
    def neighbors(self, context_id: str) -> List[str]

Description
An in-memory directed graph of context relationships.
	•	add_node(ctx)
Ensures ctx.context_id appears in the adjacency map.
	•	add_edge(from_id, to_id, label=None)
Appends to_id to the list of neighbors for from_id.
	•	neighbors(context_id) -> List[str]
Returns outgoing edges (IDs) for that node, or empty list.

⸻

Usage Example

# Initialize repository and manager
repo = ContextRepository("/path/to/contexts.jsonl")
mm   = MemoryManager(repo)

# Create a narrative context
ctx1 = ContextObject.make_narrative("Agent started", tags=["init"])
repo.save(ctx1)

# Later, recall by seed
results = mm.recall([ctx1.context_id], k=3)

# Record an additional co-activation
mm.reinforce(ctx1.context_id, [results[0].context_id])

# Prune old entries
mm.prune(ttl_hours=24)


⸻

This API lets you compose, persist, retrieve, and manage contextual artifacts in a thread-safe, versioned, and metadata-rich way—ideal for agent pipelines with memory, retrieval, and self-learning loops.