# Wave 10E Local Model Optimization

Date: 2026-06-20

## Boundary

- Olivaw only.
- No Prime Observer changes.
- No Core Signal changes.
- No new sources.
- No memory.
- No autonomy.
- No desktop automation.
- No arbitrary tool execution.

## Current Configuration

- Runtime: Ollama at `http://localhost:11434`.
- Production model: `llama3.1:8b`.
- Installed benchmark models: `llama3.1:8b`, `llama3.2:3b`.
- Recommended Ollama options added in this wave: `keep_alive="5m"`, `num_ctx=4096`, `num_predict=128`.
- Time to first token: unavailable in the current app path because Olivaw uses Ollama `/api/generate` with `stream=false`.

## Baseline Measurements

Baseline was captured with the production model before adding the Ollama runtime caps. Deterministic fast paths already bypassed model calls for source-backed, operational-unknown, and action-suggestion prompts during this measured baseline.

| Prompt | Model | Model invoked | Avg total request | Avg Ollama request | Provenance |
| --- | --- | ---: | ---: | ---: | --- |
| Network | `llama3.1:8b` | no | 892 ms | n/a | Prime Observer + Core Signal |
| Weather | `llama3.1:8b` | no | 871 ms | n/a | Weather |
| Model knowledge | `llama3.1:8b` | yes | 12,775 ms | 12,759 ms | Model knowledge |
| Operational unknown | `llama3.1:8b` | no | 0 ms | n/a | Unknown operational state |
| Action | `llama3.1:8b` | no | 0 ms | n/a | Action Registry |

The first 8B model-knowledge call included a 4,379 ms load duration. Warm 8B model-knowledge calls were 9,782 ms and 12,077 ms before output caps.

## Optimized Measurements

After adding `num_ctx=4096`, `num_predict=128`, and concise response rules:

| Prompt | `llama3.1:8b` avg | `llama3.2:3b` avg | Model invoked | Provenance |
| --- | ---: | ---: | ---: | --- |
| Network | 866 ms | 829 ms | no | Prime Observer + Core Signal |
| Weather | 878 ms | 887 ms | no | Weather |
| Model knowledge | 8,853 ms | 6,233 ms | yes | Model knowledge |
| Operational unknown | 0 ms | 0 ms | no | Unknown operational state |
| Action | 0 ms | 0 ms | no | Action Registry |

Health Review transient checks:

| Model | Attempts | Accepted | Warm latency | Quality note |
| --- | ---: | ---: | ---: | --- |
| `llama3.1:8b` | 3 | 3/3 | ~4.8 s | Best wording, conservative phrasing |
| `llama3.2:3b` | 3 | 3/3 | ~2.5 s | Faster; one answer was slightly less precise |

## Model Comparison

| Model | Footprint | Latency | Subjective quality | Provenance integrity | Recommended usage |
| --- | ---: | --- | --- | --- | --- |
| `llama3.1:8b` | 4.9 GB model, 5.3 GB loaded at 4096 ctx | Slower; warm Health Review ~4.8 s, chat knowledge ~7 s warm | Best general answer quality; less truncation risk | Preserved | Default model |
| `llama3.2:3b` | 2.0 GB model, 2.5 GB loaded at 4096 ctx | Faster; warm Health Review ~2.5 s, chat knowledge ~5.9 s warm | Good for bounded Health Review, but general knowledge still hit `num_predict` and truncated | Preserved | Fallback / Health Review candidate |

Avoid using 3B as the general default unless the chat prompt is further tightened or `num_predict` is raised for 3B, because raising the cap gives back some of its latency advantage.

## Prompt Diet Findings

- Chat system prompt after guardrail additions: 2,134 characters, about 533 tokens, 58 lines.
- Health Review system prompt: 1,254 characters, about 313 tokens, 16 lines.
- The prompt size is not the main bottleneck; generation length and model load dominate.
- Repeated assistant identity and capability lists are present on every model-backed chat call, but most production prompts now bypass the model entirely.
- Safe future reduction: collapse implemented/not-implemented capability lists into a compact capability summary while preserving the missing-capability and provenance rules.
- Do not remove source/provenance rules; they are still needed for model-knowledge prompts.

## Source Fast Path Findings

Current deterministic fast paths that avoid Ollama:

- Weather: uses `WeatherSource` directly and preserves `Source / Weather` provenance.
- Network: uses Prime Observer + Core Signal aggregate directly and preserves derived provenance.
- Operational unknowns: answer directly that no source exists, preserving unknown-operational-state provenance.
- Action requests: recognized against the existing Action Registry and suggest the action without executing it.
- Capability and source-status questions: answer from registries without model calls.

The biggest perceived-latency win is avoiding model calls, not switching models.

## Runtime Findings

- Default Ollama context was observed at 16,384 before runtime caps.
- With `num_ctx=4096`, Ollama loaded models at 4096 context.
- `keep_alive="5m"` is appropriate for a single-user M1 Mac mini: it avoids repeated cold loads without keeping models resident indefinitely.
- Do not load both 8B and 3B unnecessarily in normal production use; switching models can leave both resident for the keep-alive window.
- `num_predict=128` improves latency, but 3B still truncates some general-knowledge answers.

## Recommended Production Configuration

For an Apple M1 Mac mini with 16 GB RAM:

```toml
[providers.local]
type = "ollama"
base_url = "http://localhost:11434"
model = "llama3.1:8b"
keep_alive = "5m"
num_ctx = 4096
num_predict = 128
```

Default model: `llama3.1:8b`.

Fallback model: `llama3.2:3b`, mainly for Health Review or when memory pressure matters more than general answer quality.

Avoid: frequent model switching in the web app, because it can keep both models loaded and increase memory pressure.

## UX Latency Recommendations

- Keep source-backed and action-suggestion responses deterministic.
- Cache source aggregate results briefly within a request/page render path to avoid repeated Weather/Prime/Core reads.
- Cache model availability checks for a short TTL.
- Consider optimistic rendering for action suggestions because approval is already explicit.
- Keep Health Review cache behavior; it prevents repeated model calls on page refresh.

## Known Limitations

- Time to first token is unavailable until Olivaw uses streaming generation.
- Benchmarks were sequential and local to this Mac; thermal state and currently loaded models affect cold-start numbers.
- Weather and network source retrieval are still around 0.8-0.9 s because they read live source/provider data.
- 3B remains attractive for bounded prompts but is not reliable enough as the general default with the current response cap.
