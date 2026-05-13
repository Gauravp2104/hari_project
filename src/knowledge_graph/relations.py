"""Claude-based relation extraction with prompt caching and bounded concurrency.

Each article + its pre-extracted entities is sent to Haiku 4.5. The model returns
a list of typed relations via a strict tool-use schema. The system prompt and tool
schema are identical across all articles, so they are cached (cache_control on the
last system block also caches the tool definitions, which render before system).

Per-article responses are cached on disk (data/processed/relations_cache/) so reruns
skip already-processed articles — useful when a long batch is interrupted.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from anthropic import AsyncAnthropic, APIStatusError

from .schema import RELATION_TYPES, RELATION_DESCRIPTIONS


MODEL = "claude-haiku-4-5"
MAX_TOKENS = 2048
DEFAULT_CONCURRENCY = 5


SYSTEM_PROMPT = (
    "You are a knowledge-graph extraction assistant for the global packaging industry. "
    "You are given a packaging-news article and a list of entities (organizations, people, "
    "locations) that have already been extracted from it. Your job is to record every "
    "typed relation that the article asserts between those entities.\n\n"
    "RULES:\n"
    "1. Only use entities from the supplied list — never invent new ones. If a relation "
    "involves an entity not in the list, skip it.\n"
    "2. Use entity names exactly as they appear in the supplied list (case and spelling).\n"
    "3. Only extract relations the article EXPLICITLY supports. Do not infer or guess.\n"
    "4. Each relation must include a short evidence quote (≤ 25 words) from the article.\n"
    "5. The same pair may have multiple relations if the article supports them.\n"
    "6. Relations are directed: source acts on target.\n"
    "7. If the article asserts no relations between the listed entities, return an empty list.\n\n"
    "RELATION TYPES — use exactly one of:\n"
    + "\n".join(f"- {r}: {RELATION_DESCRIPTIONS[r]}" for r in RELATION_TYPES)
    + "\n\nEXAMPLES:\n"
    "Article: 'ProAmpac signed a definitive agreement to purchase TC Transcontinental "
    "Packaging from TC Transcontinental for US$1.51 billion.'\n"
    "Entities: ['ProAmpac', 'TC Transcontinental Packaging', 'TC Transcontinental']\n"
    '→ {"source": "ProAmpac", "target": "TC Transcontinental Packaging", "relation": "ACQUIRED", '
    '"evidence": "ProAmpac signed a definitive agreement to purchase TC Transcontinental Packaging"}\n\n'
    "Article: 'Packamama has begun manufacturing wine packaging in South Africa with local "
    "polymer producer Safripol and packaging expert Polyoak.'\n"
    "Entities: ['Packamama', 'Safripol', 'Polyoak', 'South Africa']\n"
    '→ [\n'
    '  {"source": "Packamama", "target": "Safripol", "relation": "PARTNERED_WITH", '
    '"evidence": "Packamama has begun manufacturing wine packaging ... with local polymer producer Safripol"},\n'
    '  {"source": "Packamama", "target": "Polyoak", "relation": "PARTNERED_WITH", '
    '"evidence": "with local polymer producer Safripol and packaging expert Polyoak"},\n'
    '  {"source": "Packamama", "target": "South Africa", "relation": "LOCATED_IN", '
    '"evidence": "Packamama has begun manufacturing wine packaging in South Africa"}\n'
    ']\n\n'
    "Article: 'Opportunity Green filed a complaint against ArcelorMittal, Europe's largest "
    "steel producer.'\n"
    "Entities: ['Opportunity Green', 'ArcelorMittal']\n"
    '→ {"source": "Opportunity Green", "target": "ArcelorMittal", "relation": "OPPOSES", '
    '"evidence": "Opportunity Green filed a complaint against ArcelorMittal"}\n\n'
    "Always call the extract_relations tool exactly once per article. If no relations are "
    "supported, call it with an empty list."
)


EXTRACT_TOOL = {
    "name": "extract_relations",
    "description": (
        "Record all typed relations between the listed entities that the article explicitly "
        "asserts. Each relation is a directed (source → target) edge with an evidence quote."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "relations": {
                "type": "array",
                "description": "List of typed relations extracted from the article.",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Source entity name, exactly as it appears in the supplied entity list.",
                        },
                        "target": {
                            "type": "string",
                            "description": "Target entity name, exactly as it appears in the supplied entity list.",
                        },
                        "relation": {
                            "type": "string",
                            "enum": RELATION_TYPES,
                            "description": "The typed relation between source and target.",
                        },
                        "evidence": {
                            "type": "string",
                            "description": "Short quote from the article (≤ 25 words) supporting this relation.",
                        },
                    },
                    "required": ["source", "target", "relation", "evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["relations"],
        "additionalProperties": False,
    },
}


@dataclass
class RelationExtractionResult:
    article_id: str
    relations: list[dict]
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


def _build_user_message(article_text: str, entity_names: list[str]) -> str:
    return (
        "ARTICLE:\n"
        f"{article_text.strip()}\n\n"
        "ENTITIES:\n"
        + "\n".join(f"- {n}" for n in entity_names)
        + "\n\nExtract all typed relations between these entities that the article supports."
    )


async def _extract_one(
    client: AsyncAnthropic,
    sem: asyncio.Semaphore,
    cache_dir: Path,
    article_id: str,
    article_text: str,
    entity_names: list[str],
) -> RelationExtractionResult:
    cache_path = cache_dir / f"{article_id}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            return RelationExtractionResult(article_id=article_id, **cached)
        except (json.JSONDecodeError, TypeError):
            pass

    if not entity_names or len(entity_names) < 2:
        result = RelationExtractionResult(article_id=article_id, relations=[])
        cache_path.write_text(json.dumps({"relations": []}))
        return result

    user_msg = _build_user_message(article_text, entity_names)

    async with sem:
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[EXTRACT_TOOL],
                tool_choice={"type": "tool", "name": "extract_relations"},
                messages=[{"role": "user", "content": user_msg}],
            )
        except APIStatusError as e:
            return RelationExtractionResult(
                article_id=article_id, relations=[], error=f"{e.status_code}: {e.message}"
            )
        except Exception as e:
            return RelationExtractionResult(
                article_id=article_id, relations=[], error=str(e)
            )

    relations: list[dict] = []
    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_relations":
            relations = block.input.get("relations", [])
            break

    valid_entity_names = set(entity_names)
    relations = [
        r for r in relations
        if r.get("source") in valid_entity_names
        and r.get("target") in valid_entity_names
        and r.get("source") != r.get("target")
        and r.get("relation") in RELATION_TYPES
    ]

    result = RelationExtractionResult(
        article_id=article_id,
        relations=relations,
        cache_read_tokens=response.usage.cache_read_input_tokens or 0,
        cache_creation_tokens=response.usage.cache_creation_input_tokens or 0,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )

    cache_path.write_text(json.dumps({
        "relations": relations,
        "cache_read_tokens": result.cache_read_tokens,
        "cache_creation_tokens": result.cache_creation_tokens,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }))
    return result


async def extract_all(
    jobs: list[tuple[str, str, list[str]]],
    cache_dir: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress_every: int = 25,
) -> list[RelationExtractionResult]:
    """Extract relations for many articles concurrently.

    `jobs` is a list of (article_id, article_text, [entity_name, ...]) tuples.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)

    results: list[RelationExtractionResult] = []
    tasks = [
        asyncio.create_task(_extract_one(client, sem, cache_dir, aid, text, ents))
        for aid, text, ents in jobs
    ]
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        result = await fut
        results.append(result)
        if i % progress_every == 0 or i == len(tasks):
            cache_hits = sum(1 for r in results if r.cache_read_tokens > 0)
            errors = sum(1 for r in results if r.error)
            print(
                f"  [{i}/{len(tasks)}] processed | cache hits: {cache_hits} | errors: {errors}",
                flush=True,
            )
    return results
