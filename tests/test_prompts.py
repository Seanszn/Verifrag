from src.generation.prompts import build_rag_legal_prompt


def test_build_rag_prompt_includes_structured_case_posture_constraints():
    prompt = build_rag_legal_prompt(
        "In Burnett v. United States, what did the Supreme Court decide?",
        ["Case name: Burnett v. United States\nThe petition for a writ of certiorari is denied."],
        case_posture={
            "target_case": "Burnett v. United States",
            "decision_type": "cert_denial",
            "court_action": "denied certiorari",
            "opinion_role": "dissent_from_denial",
            "author": "Gorsuch",
            "is_separate_opinion": True,
        },
    )

    assert "Structured case posture derived from the retrieved context:" in prompt
    assert "Decision type: cert denial" in prompt
    assert "Court action: denied certiorari" in prompt
    assert "Opinion author: Gorsuch" in prompt
    assert "Constraint: do not describe the separate opinion as the Court's holding." in prompt
    assert "For named-case queries, answer only with propositions explicitly supported by the retrieved text." in prompt
    assert 'Do not infer procedural posture from a caption or from the phrase "certiorari to" alone.' in prompt
    assert 'Do not say "denied certiorari" unless the retrieved text explicitly says the petition for certiorari was denied.' in prompt
    assert 'If the retrieved text contains a syllabus holding, a "Held:" sentence, or an opinion sentence stating what the Court did, prefer that language over broader paraphrase.' in prompt
    assert "Only say \"Insufficient support in retrieved authorities\" if you cannot state" in prompt
    assert "Do not use bullets or numbered lists." in prompt


def test_build_rag_prompt_allows_hyphen_bullets_in_detailed_mode():
    prompt = build_rag_legal_prompt(
        "In Esteras v. United States, explain the rule in two paragraphs and bullets.",
        ["Allowed answer fact: District courts cannot consider section 3553(a)(2)(A)."],
        response_depth="detailed",
    )

    assert "Then provide 2 to 4 short paragraphs or a compact hyphen-bullet list" in prompt
    assert "If using bullets, use simple hyphen bullets only." in prompt
    assert "Do not use bullets or numbered lists." not in prompt
