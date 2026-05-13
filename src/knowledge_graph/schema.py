"""Node and edge type definitions for the packaging-developments KG."""

from __future__ import annotations


class NodeType:
    ARTICLE = "Article"
    CATEGORY = "Category"
    ORG = "Organization"
    PERSON = "Person"
    LOCATION = "Location"


SPACY_LABEL_TO_NODE_TYPE = {
    "ORG": NodeType.ORG,
    "PERSON": NodeType.PERSON,
    "GPE": NodeType.LOCATION,
    "LOC": NodeType.LOCATION,
    "NORP": NodeType.ORG,
}


class EdgeType:
    BELONGS_TO = "BELONGS_TO"
    MENTIONS = "MENTIONS"

    ACQUIRED = "ACQUIRED"
    PARTNERED_WITH = "PARTNERED_WITH"
    INVESTED_IN = "INVESTED_IN"
    SUPPLIES = "SUPPLIES"
    PRODUCES = "PRODUCES"
    REGULATES = "REGULATES"
    SUBJECT_TO = "SUBJECT_TO"
    EMPLOYED_BY = "EMPLOYED_BY"
    LOCATED_IN = "LOCATED_IN"
    OPPOSES = "OPPOSES"
    COMPETES_WITH = "COMPETES_WITH"
    COLLABORATES_WITH = "COLLABORATES_WITH"


RELATION_TYPES = [
    EdgeType.ACQUIRED,
    EdgeType.PARTNERED_WITH,
    EdgeType.INVESTED_IN,
    EdgeType.SUPPLIES,
    EdgeType.PRODUCES,
    EdgeType.REGULATES,
    EdgeType.SUBJECT_TO,
    EdgeType.EMPLOYED_BY,
    EdgeType.LOCATED_IN,
    EdgeType.OPPOSES,
    EdgeType.COMPETES_WITH,
    EdgeType.COLLABORATES_WITH,
]

RELATION_DESCRIPTIONS = {
    EdgeType.ACQUIRED: "Source company acquired/bought target company.",
    EdgeType.PARTNERED_WITH: "Source and target formed a formal partnership or joint venture.",
    EdgeType.INVESTED_IN: "Source invested money or equity in target.",
    EdgeType.SUPPLIES: "Source supplies materials, components, or services to target.",
    EdgeType.PRODUCES: "Source company produces target product/material.",
    EdgeType.REGULATES: "Source (regulator/government) regulates target industry/company.",
    EdgeType.SUBJECT_TO: "Source company is subject to target regulation/policy/law.",
    EdgeType.EMPLOYED_BY: "Source person is employed by/holds a role at target organization.",
    EdgeType.LOCATED_IN: "Source organization is headquartered or operates in target location.",
    EdgeType.OPPOSES: "Source filed a complaint, lawsuit, or public opposition against target.",
    EdgeType.COMPETES_WITH: "Source and target are direct market competitors.",
    EdgeType.COLLABORATES_WITH: "Source and target are jointly working on a project or initiative (lighter than partnership).",
}
