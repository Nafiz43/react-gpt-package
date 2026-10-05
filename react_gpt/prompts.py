"""Research prompts and definitions, preserved from the original pipeline.

Copyright © 2025 The Regents of the University of California, Davis campus.
All Rights Reserved. Used with permission.
"""


prompt_template = """
You are an expert systematic literature reviewer specializing in Open Source Software (OSS) sustainability research.

Your task is to extract ACTIONABLE RECOMMENDATIONS from the article text provided below.

STRICT DEFINITION — An actionable recommendation MUST satisfy ALL of the following:
  1. It is a concrete, specific action that a person or team can directly implement.
  2. It is expressed as an imperative or directive (e.g., "use X", "adopt Y", "implement Z").
  3. It is distinct from a general observation, finding, or description of current practice.
  4. It pertains to improving the sustainability, contribution, or maintenance of OSS projects.

DO NOT extract:
  - General observations or descriptions (e.g., "projects tend to have more bugs")
  - Vague suggestions without a clear action (e.g., "more work is needed")
  - Future research directions (e.g., "future studies should explore X")
  - Statements about what the authors did (e.g., "we analyzed 100 projects")

EXTRACTION RULES:
  - Extract ONLY recommendations that are explicitly stated or very directly implied in the text.
  - Do NOT infer, speculate, or generalize beyond what is written.
  - Each recommendation must be a single, self-contained action — split compound recommendations.
  - Keep each field concise: 1–3 sentences maximum per field.
  - For 'positive_impact': quote or closely paraphrase only what the article explicitly states will improve.
  - For 'evidence': cite the specific data, experiment, or finding in the article that justifies the recommendation. Use exact figures or results where available.
  - For 'confidence': score 0.9–1.0 only if the article explicitly states the recommendation with clear evidence; 0.7–0.89 if implied with some evidence; below 0.7 if weakly supported.

OUTPUT: Return ONLY a valid JSON object. No preamble, no explanation, no markdown fences.

If no actionable recommendations exist in the text, return exactly:
{"message": "NO ACTIONABLE CAN BE DERIVED"}

Full text of the article:
"""

IP_template = """
Using the definition and rules provided above, extract all actionable recommendations from the article.

Return ONLY this JSON structure, with no additional text:
{
  "recommendations": [
    {
      "recommendation": "<single concrete action, 1-2 sentences>",
      "positive_impact": "<specific improvement stated in the article, or 'NO IMPACT FOUND'>",
      "evidence": "<specific data point, result, or finding from the article, or 'NO EVIDENCE FOUND'>",
      "confidence": <0.0 to 1.0>
    }
  ]
}
"""

CoT_template = """
Using the definition and rules provided above, extract actionable recommendations by working through these steps internally before producing output:

STEP 1 — SCAN: Read every sentence. Flag only sentences containing imperative directives or explicit suggestions targeted at practitioners.
STEP 2 — FILTER: Discard any flagged sentence that is a general observation, future work direction, or lacks a specific implementable action.
STEP 3 — DEDUPLICATE: Merge recommendations that describe the same action even if worded differently.
STEP 4 — ENRICH: For each surviving recommendation, locate the specific impact claim and supporting evidence in the article text. Use exact numbers or findings where present.
STEP 5 — SCORE: Assign confidence based strictly on explicitness (0.9–1.0 = explicitly stated with evidence, 0.7–0.89 = implied with partial evidence, <0.7 = weakly supported).

Return ONLY this JSON structure, with no additional text:
{
  "recommendations": [
    {
      "recommendation": "<single concrete action, 1-2 sentences>",
      "positive_impact": "<specific improvement stated in the article, or 'NO IMPACT FOUND'>",
      "evidence": "<specific data point, result, or finding from the article, or 'NO EVIDENCE FOUND'>",
      "confidence": <0.0 to 1.0>
    }
  ]
}
"""

RA_template = """
Using the definition and rules provided above, extract actionable recommendations using this retrieval-augmented reasoning approach:

For EACH candidate recommendation:

  [CHECK 1 — ACTIONABILITY]
  Is this a concrete, implementable action (not an observation or future work)?
  → If NO: discard. If YES: continue.

  [CHECK 2 — DIRECTNESS]
  Is this recommendation explicitly stated in the article (not inferred by you)?
  → If NO: discard. If YES: continue.

  [CHECK 3 — IMPACT]
  Does the article explicitly state what will improve if this is adopted?
  → If YES: extract the exact claim. If NO: set "NO IMPACT FOUND".

  [CHECK 4 — EVIDENCE]
  Does the article provide a specific data point, study result, or empirical finding supporting this recommendation?
  → If YES: extract it verbatim or near-verbatim. If NO: set "NO EVIDENCE FOUND".

  [CHECK 5 — CONFIDENCE]
  Rate confidence 0.9–1.0 (explicit + evidence), 0.7–0.89 (implied + partial evidence), <0.7 (weak).

Return ONLY this JSON structure, with no additional text:
{
  "recommendations": [
    {
      "recommendation": "<single concrete action, 1-2 sentences>",
      "positive_impact": "<specific improvement stated in the article, or 'NO IMPACT FOUND'>",
      "evidence": "<specific data point, result, or finding from the article, or 'NO EVIDENCE FOUND'>",
      "confidence": <0.0 to 1.0>
    }
  ]
}
"""

CATEGORIES = {
    "New Contributor Onboarding and Involvement": (
        "This category focuses on ensuring that new contributors can easily join, understand, "
        "and meaningfully contribute to the project. "
        "Criteria: (a) Actionable facilitates the integration of new contributors by providing "
        "mentorship, onboarding materials, or simplifying the contribution process; "
        "(b) Actionable relates to improving project documentation or offering better support "
        "mechanisms for first-time contributors; "
        "(c) Actionable helps build a welcoming, inclusive, and open culture for new participants."
    ),
    "Code Standards and Maintainability": (
        "This category deals with ensuring that the codebase adheres to established standards, "
        "making it easier to maintain and scale. It includes efforts to ensure code readability, "
        "modularity, and compliance with coding best practices. "
        "Criteria: (a) Actionable relates to improving the quality, readability, or structure of "
        "the codebase; (b) Actionable includes efforts to enforce coding guidelines, refactor code "
        "for better maintainability, or reduce technical debt; "
        "(c) Actionable includes the use of linters, formatters, or static code analysis tools."
    ),
    "Automated Testing and Quality Assurance": (
        "This category focuses on ensuring the project's robustness and reliability through "
        "automated testing practices, such as unit, integration, and end-to-end tests. It also "
        "includes broader quality assurance activities. "
        "Criteria: (a) Actionable involves the implementation or improvement of automated testing "
        "frameworks and testing strategies; (b) Actionable includes practices that ensure the "
        "detection of bugs early in the development cycle and ensure high-quality releases."
    ),
    "Community Collaboration and Engagement": (
        "This category deals with activities that foster collaboration, communication, and "
        "engagement within the OSS community. It includes practices for keeping the community "
        "active and involved. "
        "Criteria: (a) Actionable aims to improve communication between contributors, maintainers, "
        "and users; (b) Actionable involves organizing community-driven events, discussions, or "
        "collaborations, as well as platforms to enhance transparency and teamwork; "
        "(c) Actionable relates to tools and processes for better community governance and "
        "decision-making."
    ),
    "Documentation Practices": (
        "This category focuses on ensuring that the project's documentation is thorough, "
        "up-to-date, and easily accessible. Documentation practices are crucial for both current "
        "and future contributors. "
        "Criteria: (a) Actionable focuses on improving the quality, clarity, or accessibility of "
        "project documentation, such as user guides, API references, or contributor guides; "
        "(b) Actionable includes practices for keeping documentation synchronized with the "
        "codebase and ensuring it meets the needs of different stakeholders; "
        "(c) Actionable involves translation efforts or making documentation more accessible to "
        "non-expert audiences."
    ),
    "Project Management and Governance": (
        "This category deals with the governance structure and project management practices that "
        "keep the project organized, transparent, and sustainable over the long term. "
        "Criteria: (a) Actionable enhances the governance model, clarifies roles and "
        "responsibilities, or improves the decision-making process; (b) Actionable involves "
        "defining or refining processes for issue triaging, release management, or conflict "
        "resolution; (c) Actionable includes efforts to improve the transparency of project "
        "goals, progress, and decision-making."
    ),
    "Security Best Practices and Legal Compliance": (
        "This category addresses efforts to secure the project and ensure compliance with relevant "
        "legal standards, such as licenses, data privacy laws, and security protocols. "
        "Criteria: (a) Actionable focuses on improving the security posture of the project by "
        "following best practices, addressing vulnerabilities, or conducting audits; "
        "(b) Actionable involves ensuring compliance with open-source licenses, setting up "
        "contributor license agreements (CLAs), or aligning with data privacy regulations; "
        "(c) Actionable includes security measures such as dependency management, security audits, "
        "and secure coding practices."
    ),
    "CI/CD and DevOps Automation": (
        "This category deals with continuous integration and continuous deployment (CI/CD) "
        "processes that automate building, testing, and deployment pipelines. It also includes "
        "broader DevOps automation tasks. "
        "Criteria: (a) Actionable involves the setup or enhancement of CI/CD pipelines to ensure "
        "faster, reliable, and automated releases; (b) Actionable relates to automating "
        "infrastructure provisioning, containerization, or deployment to cloud environments; "
        "(c) Actionable includes the integration of DevOps practices that ensure smooth, "
        "automated, and repeatable processes for software development, testing, and deployment."
    ),
}

SOUND_DEFINITION = (
    "A ReACT is SOUND if it makes logical sense, has no contradictions, "
    "and all parts of the recommendation work together consistently. "
    "For instance, 'Project Managers should use peer reviews and automated tools "
    "for code review' is SOUND because peer reviews and tools work together to "
    "improve code quality. On the other hand, 'To improve user satisfaction, add "
    "new features without onboarding newcomers' is UNSOUND because adding features "
    "without testing can hurt user satisfaction, and not onboarding newcomers can "
    "reduce support quality."
)

PRECISE_DEFINITION = (
    "A ReACT is PRECISE if it is clear, specific, easy to follow, and leaves no "
    "room for confusion. For example, 'To attract newcomers, help them make their "
    "first contribution' is PRECISE as it gives a clear action to take. On the "
    "contrary, 'To attract core developers, ensure high code quality' is IMPRECISE "
    "as it does not explain HOW to ensure high code quality."
)

FEATURE_PROMPT = """
You are given an actionable recommendation related to OSS project sustainability.

Your task is to identify which features from the list below would likely be impacted by this actionable.

Each feature captures a specific aspect of socio-technical activity in open-source software projects:

- "s_avg_clustering_coef": Measures how interconnected a developer's immediate social contacts are within the social network. Higher values indicate tighter community cohesion.
- "s_net_overlap": Number of developers who remain consistently active in the social network over time. Reflects long-term social engagement.
- "t_num_dev_nodes": Number of unique developers participating in technical contributions during a given period. Indicates the scale of technical involvement.
- "t_num_dev_per_file": Average number of developers contributing to each file. Reflects collaborative intensity and shared ownership of code.
- "t_graph_density": Density of the technical network. Higher values suggest more collaboration via shared files between developers.
- "t_net_overlap": Number of developers who remain consistently active in technical work across time. Reflects technical contributor stability.
- "st_num_dev": Number of developers who are active in both social and technical networks. Indicates integration of communication and contribution—a known driver of sustainability.

Now, based on the actionable provided below, identify which features it is most likely to influence. List the feature names only, separated by commas.

ACTIONABLE:
"""


MEDICINE_CATEGORIES = {
    "Imaging and Diagnosis": "Image acquisition, interpretation, diagnostic assessment, and diagnostic workflows.",
    "Treatment and Procedures": "Therapeutic or interventional procedures in the specific population studied.",
    "Patient Safety and Risk Reduction": "Reducing radiation, complications, errors, or other patient risks.",
    "Clinical Workflow and Training": "Clinical team processes, training, supervision, and workflow integration.",
    "Data Quality and Annotation": "Dataset curation, labeling, annotations, and data quality controls.",
    "Validation and Evidence Quality": "External validation, representative evaluation, study design, and reproducibility.",
    "Privacy and Governance": "Privacy, consent, data governance, and responsible implementation."
}


def domain_categories(domain):
    return MEDICINE_CATEGORIES if domain == "medicine" else CATEGORIES


def domain_extraction(domain):
    if domain == "oss":
        return prompt_template
    return prompt_template.replace(
        "Open Source Software (OSS) sustainability research", "medical and medical-imaging research"
    ).replace(
        "It pertains to improving the sustainability, contribution, or maintenance of OSS projects.",
        "It pertains to a specific medical, clinical, medical-imaging, or medical-research practice."
    ) + """
Preserve the studied population, setting, conditions, and limitations in each recommendation.
Distinguish experimental methods from established clinical practice. Do not turn a study result
into a treatment directive or infer safety, efficacy, or generalizability beyond the article.
Extract only the authors' explicit or directly supported recommendations. A lack of eligible
recommendations is a valid result. This is a research-literature extraction, not patient advice.
"""


def domain_quality(domain):
    if domain == "oss":
        return SOUND_DEFINITION, PRECISE_DEFINITION
    return (
        "A recommendation is SOUND if it is logically coherent and its action, claimed impact, "
        "and stated evidence are consistent. Reject unsupported leaps from experimental results "
        "to general clinical adoption. This judgment does not establish clinical validity.",
        "A recommendation is PRECISE if it specifies a concrete implementable action, its target, "
        "and relevant setting or conditions. Vague aspirations and unspecified future research are not precise."
    )
