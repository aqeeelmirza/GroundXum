# KG Extraction Prompt — GroundSumm

You will extract a small knowledge graph from a single generated video summary.

## Task

Given:
- `summary`: a generated summary text from a multimodal video summarizer
- `dataset`: one of `youcook2` (cooking videos) or `videoxum` (open-domain activities)

Return a JSON object with `entities` and `triples`, extracted ONLY from what is literally stated in the summary.

## Schema

### YouCook2 (cooking)
**Entity types:** `ingredient`, `tool`, `action`, `attribute`
**Relation types:** `uses_ingredient` (action→ingredient), `uses_tool` (action→tool), `has_attribute` (ingredient→attribute), `follows` (action→action)

### VideoXum (open-domain)
**Entity types:** `person`, `object`, `setting`, `action`, `attribute`
**Relation types:** `performs` (person→action), `uses_object` (action→object), `located_in` (person|action→setting), `has_attribute` (entity→attribute)

## Rules

1. **Literal only.** Extract only what is explicitly stated. Do not infer commonsense additions.
2. **Preserve malformed tokens.** If a token looks malformed (e.g., "verdalago", "garala", "cormeric"), keep it AS-IS in the surface form. Set `"malformed": true`. Do NOT normalize to the likely intended word.
3. **Deduplicate.** If the same entity appears multiple times, extract once.
4. **Lowercase** all surface forms.
5. **Degenerate input.** If the summary is incoherent or contains no extractable entities, return empty lists and set `"degenerate": true`.

## Output format

Return ONLY valid JSON, no prose, no markdown fences:

```json
{
  "summary_id": "<echo the id provided>",
  "dataset": "youcook2 | videoxum",
  "degenerate": false,
  "entities": [
    {"id": "e1", "type": "ingredient", "surface": "garlic", "malformed": false},
    {"id": "e2", "type": "action", "surface": "saute", "malformed": false}
  ],
  "triples": [
    {"head": "e2", "relation": "uses_ingredient", "tail": "e1"}
  ]
}
```

## Examples

### Example 1 (YouCook2, clean)
Summary: "Prepare a flavorful beef stew by cooking onions with garlic, spices, and flour, then mix with tomato puree."

Output:
```json
{
  "summary_id": "yc2_001",
  "dataset": "youcook2",
  "degenerate": false,
  "entities": [
    {"id": "e1", "type": "ingredient", "surface": "beef stew", "malformed": false},
    {"id": "e2", "type": "ingredient", "surface": "onions", "malformed": false},
    {"id": "e3", "type": "ingredient", "surface": "garlic", "malformed": false},
    {"id": "e4", "type": "ingredient", "surface": "spices", "malformed": false},
    {"id": "e5", "type": "ingredient", "surface": "flour", "malformed": false},
    {"id": "e6", "type": "ingredient", "surface": "tomato puree", "malformed": false},
    {"id": "e7", "type": "action", "surface": "cook", "malformed": false},
    {"id": "e8", "type": "action", "surface": "mix", "malformed": false},
    {"id": "e9", "type": "attribute", "surface": "flavorful", "malformed": false}
  ],
  "triples": [
    {"head": "e7", "relation": "uses_ingredient", "tail": "e2"},
    {"head": "e7", "relation": "uses_ingredient", "tail": "e3"},
    {"head": "e7", "relation": "uses_ingredient", "tail": "e4"},
    {"head": "e7", "relation": "uses_ingredient", "tail": "e5"},
    {"head": "e8", "relation": "uses_ingredient", "tail": "e6"},
    {"head": "e7", "relation": "follows", "tail": "e8"},
    {"head": "e1", "relation": "has_attribute", "tail": "e9"}
  ]
}
```

### Example 2 (YouCook2, with malformed token)
Summary: "Make a flavorful lamb masala by sauteing garala with garlic, ginger, and onions in oil."

Output:
```json
{
  "summary_id": "yc2_005",
  "dataset": "youcook2",
  "degenerate": false,
  "entities": [
    {"id": "e1", "type": "ingredient", "surface": "lamb masala", "malformed": false},
    {"id": "e2", "type": "ingredient", "surface": "garala", "malformed": true},
    {"id": "e3", "type": "ingredient", "surface": "garlic", "malformed": false},
    {"id": "e4", "type": "ingredient", "surface": "ginger", "malformed": false},
    {"id": "e5", "type": "ingredient", "surface": "onions", "malformed": false},
    {"id": "e6", "type": "ingredient", "surface": "oil", "malformed": false},
    {"id": "e7", "type": "action", "surface": "make", "malformed": false},
    {"id": "e8", "type": "action", "surface": "saute", "malformed": false},
    {"id": "e9", "type": "attribute", "surface": "flavorful", "malformed": false}
  ],
  "triples": [
    {"head": "e8", "relation": "uses_ingredient", "tail": "e2"},
    {"head": "e8", "relation": "uses_ingredient", "tail": "e3"},
    {"head": "e8", "relation": "uses_ingredient", "tail": "e4"},
    {"head": "e8", "relation": "uses_ingredient", "tail": "e5"},
    {"head": "e8", "relation": "uses_ingredient", "tail": "e6"},
    {"head": "e1", "relation": "has_attribute", "tail": "e9"}
  ]
}
```

### Example 3 (VideoXum, clean)
Summary: "A group of hockey players are skating around on the ice. They are playing a game of hockey."

Output:
```json
{
  "summary_id": "vx_006",
  "dataset": "videoxum",
  "degenerate": false,
  "entities": [
    {"id": "e1", "type": "person", "surface": "group of hockey players", "malformed": false},
    {"id": "e2", "type": "setting", "surface": "ice", "malformed": false},
    {"id": "e3", "type": "object", "surface": "hockey", "malformed": false},
    {"id": "e4", "type": "action", "surface": "skate", "malformed": false},
    {"id": "e5", "type": "action", "surface": "play", "malformed": false}
  ],
  "triples": [
    {"head": "e1", "relation": "performs", "tail": "e4"},
    {"head": "e1", "relation": "performs", "tail": "e5"},
    {"head": "e4", "relation": "located_in", "tail": "e2"},
    {"head": "e5", "relation": "uses_object", "tail": "e3"}
  ]
}
```

### Example 4 (degenerate, COIN-style — kept as edge case)
Summary: "video demonstrates MakeCCC by by the the floor floor, op use use use, and and floor floor floor."

Output:
```json
{
  "summary_id": "degen_001",
  "dataset": "youcook2",
  "degenerate": true,
  "entities": [],
  "triples": []
}
```
