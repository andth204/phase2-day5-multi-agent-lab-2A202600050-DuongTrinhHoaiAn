## Problem
Xây hệ thống research assistant tự động: nhận câu hỏi kỹ thuật dài,
tìm kiếm nguồn, phân tích, và viết câu trả lời ~500 từ có trích dẫn.

## Why multi-agent?
Single-agent gộp search + analysis + writing vào 1 prompt → kết quả
kém hơn, không audit được từng bước, không có quality scoring.

## Agent roles
| Agent      | Responsibility        | Input                        | Output           | Failure mode                  |
|------------|-----------------------|------------------------------|------------------|-------------------------------|
| Supervisor | Điều phối routing     | ResearchState                | route_history    | Vòng lặp vô hạn → max_iter   |
| Researcher | Search + summarise    | query, max_sources           | research_notes   | Mock search → hallucination   |
| Analyst    | Phân tích evidence    | research_notes               | analysis_notes   | Weak notes → weak analysis    |
| Writer     | Viết final answer     | research_notes, analysis     | final_answer     | Verbose nếu notes quá dài     |
| Critic     | Chấm điểm quality     | research_notes, final_answer | quality_score    | Over-generous scoring         |

## Routing policy
researcher → analyst → writer → done
Guardrail: nếu iteration >= max_iterations → force done

## Guardrails
- Max iterations: 6
- Timeout: 60s (OpenAI call)
- Retry: tenacity 3 lần, exponential backoff
- Fallback: mock search nếu không có Tavily key
- Validation: Pydantic schema cho mọi input/output

## Benchmark plan
Query: "Research GraphRAG state-of-the-art and write a 500-word summary"
→ Kết quả: single 16s/$0.0004 vs multi 33s/$0.0017, quality 8.0/10