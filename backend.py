import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from logits import _compute_all_components_logits
from model import get_cache, get_output
from prompt import check_answer_correctness, get_prompt_samples, get_random_prompt


MODEL_NAME = "gpt2-small"

app = FastAPI(title="Visualize LLM API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: HookedTransformer | None = None
_model_lock = threading.Lock()
_jobs: dict[str, "AnalysisJob"] = {}
_jobs_lock = threading.Lock()


class AnalyzeRequest(BaseModel):
    prompt: str
    expected_answer: str = ""


class AnalyzeResponse(BaseModel):
    job_id: str


class RandomPromptResponse(BaseModel):
    prompt: str
    expected_answer: str


class PromptSampleResponse(BaseModel):
    prompt: str
    subject: str
    expected_answer: str
    keywords: str


class NodeStatus(BaseModel):
    kind: Literal["input", "attention", "mlp", "output"]
    ready: bool
    rank: int | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    state: Literal["queued", "running", "completed", "failed"]
    prompt: str
    expected_answer: str
    output: str | None
    is_correct: bool | None
    n_layers: int | None
    n_heads: int | None
    completed_nodes: int
    total_nodes: int | None
    current_step: str
    error: str | None
    nodes: dict[str, NodeStatus]


class AttentionData(BaseModel):
    tokens: list[str]
    values: list[list[float]]


class LogitsData(BaseModel):
    tokens: list[str]
    values: list[float]


class NodeDetailResponse(BaseModel):
    node: str
    kind: Literal["input", "attention", "mlp", "output"]
    rank: int | None
    attention: AttentionData | None = None
    logits: LogitsData | None = None


@dataclass
class NodeRecord:
    kind: Literal["input", "attention", "mlp", "output"]
    ready: bool = False
    rank: int | None = None
    attention: AttentionData | None = None
    logits: LogitsData | None = None


@dataclass
class AnalysisJob:
    job_id: str
    prompt: str
    expected_answer: str
    state: Literal["queued", "running", "completed", "failed"] = "queued"
    output: str | None = None
    is_correct: bool | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    current_step: str = "待機中"
    error: str | None = None
    nodes: dict[str, NodeRecord] = field(default_factory=dict)

    @property
    def total_nodes(self) -> int | None:
        if self.n_layers is None or self.n_heads is None:
            return None
        return 1 + self.n_layers * (self.n_heads + 1) + 1

    @property
    def completed_nodes(self) -> int:
        return sum(1 for node in self.nodes.values() if node.ready)


def load_model() -> HookedTransformer:
    global _model
    with _model_lock:
        if _model is None:
            _model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")
        return _model


def snapshot_job(job_id: str) -> AnalysisJob:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job


def init_nodes(job: AnalysisJob, ranks: dict[str, int], n_layers: int, n_heads: int) -> None:
    nodes: dict[str, NodeRecord] = {
        "Input": NodeRecord(kind="input", ready=True, rank=ranks.get("Input"))
    }
    for layer in range(n_layers):
        for head in range(n_heads):
            node = f"A{layer}.H{head}"
            nodes[node] = NodeRecord(kind="attention", rank=ranks.get(node))
        node = f"MLP{layer}"
        nodes[node] = NodeRecord(kind="mlp", rank=ranks.get(node))
    nodes["Output"] = NodeRecord(kind="output", rank=ranks.get("Output"))
    job.nodes = nodes


def mark_node_ready(
    job_id: str,
    node_name: str,
    attention: AttentionData | None = None,
    logits: LogitsData | None = None,
) -> None:
    with _jobs_lock:
        node = _jobs[job_id].nodes[node_name]
        node.attention = attention
        node.logits = logits
        node.ready = True


def update_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        for key, value in fields.items():
            setattr(job, key, value)


def run_analysis(job_id: str) -> None:
    try:
        update_job(job_id, state="running", current_step="モデルを読み込み中")
        model = load_model()
        job = snapshot_job(job_id)

        update_job(job_id, current_step="入力を解析中")
        logits, cache = get_cache(model, job.prompt)
        output = get_output(model, logits)
        is_correct = check_answer_correctness(
            model, job.prompt, logits, job.expected_answer
        )

        update_job(job_id, current_step="ノードの順位を計算中")
        layer_logits, head_logits = _compute_all_components_logits(model, cache)
        ranks = calculate_ranks(model, job.prompt, job.expected_answer, layer_logits, head_logits)

        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        with _jobs_lock:
            current = _jobs[job_id]
            current.output = output
            current.is_correct = is_correct
            current.n_layers = n_layers
            current.n_heads = n_heads
            init_nodes(current, ranks, n_layers, n_heads)

        tokens = [token.replace(" ", "_") for token in model.to_str_tokens(job.prompt, prepend_bos=False)]

        update_job(
            job_id,
            current_step="各データを生成中（明るくなったところはクリックできます）",
        )
        for layer in reversed(range(n_layers)):
            layer_detail = top_logits_data(layer_logits[layer], model)
            mark_node_ready(job_id, f"MLP{layer}", logits=layer_detail)

            if layer == n_layers - 1:
                mark_node_ready(job_id, "Output", logits=layer_detail)

        for layer in reversed(range(n_layers)):
            layer_key = get_act_name("attn", layer)
            layer_attn = cache[layer_key][0].detach().cpu().numpy()

            for head in range(n_heads):
                mark_node_ready(
                    job_id,
                    f"A{layer}.H{head}",
                    attention=AttentionData(
                        tokens=tokens,
                        values=round_matrix(layer_attn[head]),
                    ),
                    logits=top_logits_data(head_logits[layer, head], model),
                )

        update_job(job_id, state="completed", current_step="予測完了")
    except Exception as exc:
        update_job(job_id, state="failed", current_step="失敗", error=str(exc))


def round_matrix(values, digits: int = 4) -> list[list[float]]:
    return [[round(float(value), digits) for value in row] for row in values]


def top_logits_data(
    values: torch.Tensor, model: HookedTransformer, top_k: int = 10
) -> LogitsData:
    top_values, top_indices = torch.topk(values.detach().cpu(), k=top_k)
    tokens = [model.tokenizer.decode([int(index)]).replace(" ", "_") for index in top_indices]
    return LogitsData(
        tokens=tokens,
        values=[round(float(value), 4) for value in top_values],
    )


def calculate_ranks(
    model: HookedTransformer,
    prompt: str,
    expected_answer: str,
    layer_logits: torch.Tensor,
    head_logits: torch.Tensor,
) -> dict[str, int]:
    if not expected_answer:
        return {}

    full_text_with_space = prompt + " " + expected_answer
    full_text_without_space = prompt + expected_answer
    prompt_tokens = model.to_tokens(prompt, prepend_bos=False)[0]
    prompt_length = len(prompt_tokens)
    full_tokens_with_space = model.to_tokens(full_text_with_space, prepend_bos=False)[0]
    full_tokens_without_space = model.to_tokens(full_text_without_space, prepend_bos=False)[0]

    object_token_id = None
    if len(full_tokens_with_space) > prompt_length:
        object_token_id = full_tokens_with_space[prompt_length].item()
    elif len(full_tokens_without_space) > prompt_length:
        object_token_id = full_tokens_without_space[prompt_length].item()
    if object_token_id is None:
        return {}

    ranks: dict[str, int] = {"Input": model.cfg.d_vocab}
    for layer_idx in range(model.cfg.n_layers):
        sorted_indices = torch.argsort(layer_logits[layer_idx], descending=True)
        ranks[f"MLP{layer_idx}"] = (
            (sorted_indices == object_token_id).nonzero(as_tuple=True)[0].item() + 1
        )
        for head_idx in range(model.cfg.n_heads):
            sorted_indices = torch.argsort(head_logits[layer_idx, head_idx], descending=True)
            ranks[f"A{layer_idx}.H{head_idx}"] = (
                (sorted_indices == object_token_id).nonzero(as_tuple=True)[0].item() + 1
            )
    ranks["Output"] = ranks[f"MLP{model.cfg.n_layers - 1}"]
    return ranks


def serialize_job(job: AnalysisJob) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        state=job.state,
        prompt=job.prompt,
        expected_answer=job.expected_answer,
        output=job.output,
        is_correct=job.is_correct,
        n_layers=job.n_layers,
        n_heads=job.n_heads,
        completed_nodes=job.completed_nodes,
        total_nodes=job.total_nodes,
        current_step=job.current_step,
        error=job.error,
        nodes={
            name: NodeStatus(kind=node.kind, ready=node.ready, rank=node.rank)
            for name, node in job.nodes.items()
        },
    )


@app.get("/api/random-prompt", response_model=RandomPromptResponse)
def random_prompt() -> RandomPromptResponse:
    prompt, expected_answer = get_random_prompt()
    return RandomPromptResponse(prompt=prompt, expected_answer=expected_answer)


@app.get("/api/prompt-samples", response_model=list[PromptSampleResponse])
def prompt_samples() -> list[PromptSampleResponse]:
    return [PromptSampleResponse(**sample) for sample in get_prompt_samples()]


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    job_id = uuid.uuid4().hex
    job = AnalysisJob(
        job_id=job_id,
        prompt=prompt,
        expected_answer=request.expected_answer.strip(),
    )
    with _jobs_lock:
        _jobs[job_id] = job

    thread = threading.Thread(target=run_analysis, args=(job_id,), daemon=True)
    thread.start()
    return AnalyzeResponse(job_id=job_id)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    return serialize_job(snapshot_job(job_id))


@app.get("/api/jobs/{job_id}/nodes/{node_name}", response_model=NodeDetailResponse)
def node_detail(job_id: str, node_name: str) -> NodeDetailResponse:
    job = snapshot_job(job_id)
    node = job.nodes.get(node_name)
    if node is None:
        raise HTTPException(status_code=404, detail="node not found")
    if not node.ready:
        raise HTTPException(status_code=202, detail="node is not ready")
    return NodeDetailResponse(
        node=node_name,
        kind=node.kind,
        rank=node.rank,
        attention=node.attention,
        logits=node.logits,
    )


# Serve the built frontend from the same origin (production). Mounted after all
# /api routes so those keep priority; guarded so dev (no dist yet) still works.
_DIST = Path(__file__).parent / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
