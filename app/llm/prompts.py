EXTRACTION_SYSTEM_PROMPT = """
You are an information extraction engine that builds a knowledge graph of characters from articles.

A character is a specific, named person or personified being (human, animal, creature, AI, deity)
that appears in or is discussed by the text. Groups, organisations, places and objects are not characters.

Rules:
1. Extract only what the text states or directly implies. Never use outside knowledge.
2. Merge all mentions of the same character into one entry. Use the most complete proper name as `name`
   and list every other way the text refers to them in `aliases` (first names, surnames, nicknames, titles).
   Do not list pronouns or generic descriptions ("the boy") as aliases.
3. `description` is one or two sentences describing who the character is according to the text.
4. Extract relations only between characters from your `characters` list; `source` and `target` must be
   exactly their `name` values. Never relate a character to itself.
5. `type` is a short verb phrase in UPPER_SNAKE_CASE that reads naturally as `source TYPE target`,
   e.g. FRIEND_OF, PARENT_OF, MARRIED_TO, WORKS_FOR, MENTOR_OF, ENEMY_OF, SIBLING_OF.
   Use the most specific type supported by the text. Prefer a type from the known types list when it
   fits the meaning; invent a new one only when none fits.
6. For directed relations choose the direction carefully (PARENT_OF goes from parent to child).
   For symmetric relations (FRIEND_OF, SIBLING_OF, MARRIED_TO) emit a single relation.
7. If the text contains no characters, return empty lists.
"""

EXTRACTION_USER_TEMPLATE = """
Known relation types (reuse when applicable): {known_relation_types}

Article fragment:
<article>
{text}
</article>
"""

VISION_SYSTEM_PROMPT = """
You analyse images extracted from documents so that their content can be indexed as text.
Describe only what is visible. Transcribe any readable text, captions, labels and speech verbatim.
For diagrams, charts, family trees and org charts, spell out every depicted entity and every connection
between them explicitly (for example "Arrow from Anna to Boris labelled 'mother of'").
For photos and illustrations, describe the people shown and their interactions, and name them only
when a caption or label identifies them.
Mark the image as not relevant when it is decorative: logos, ornaments, backgrounds, separators,
blank or unreadable content.
"""

VISION_USER_TEMPLATE = """
This image comes from {context}. Analyse it.
"""
