"""
LSF Domain-Specific Prompt Template for LinkNER LLM Classification
Contains clean, leakage-free Definitions & Guidelines (D&G) based on the SLIMER framework.
All guidelines are high-level conceptual definitions with ZERO test-set entity leakage.
Supports both legacy and relabeled ontology schemas via use_new_labels parameter.
"""

from typing import Dict, Any, List

# Legacy Category Definitions & Guidelines
LSF_CATEGORY_DEFINITIONS_LEGACY: Dict[str, Dict[str, str]] = {
    "Beauty_and_Cleaning": {
        "definition": "Products, practices, and personal care items related to body grooming, cosmetics, hygiene, and environmental sanitation.",
        "includes": "Dental care items, skin care, cosmetics, personal care products, disinfectants, and household cleaning agents.",
        "excludes": "Do NOT label medical surgical procedures or industrial pollution exposures."
    },
    "Drugs": {
        "definition": "Substances, medications, or addictive chemical agents consumed for therapeutic or psychoactive effects.",
        "includes": "Tobacco products, nicotine, alcoholic beverages, illicit substances, prescription/over-the-counter pharmaceuticals, and placebos.",
        "excludes": "Do NOT label essential food items or dietary nutrients."
    },
    "Environmental_exposures": {
        "definition": "External physical, chemical, or radiological agents and conditions present in workplace, ambient, or indoor air.",
        "includes": "Airborne contaminants, ventilation conditions, heavy metal toxicants, radiation, noise, and environmental pollutants.",
        "excludes": "Do NOT label personal voluntary habits like dietary intake or voluntary substance use."
    },
    "Lifestyle_factor": {
        "definition": "Composite or aggregate health habit indicators representing overall lifestyle behavior patterns.",
        "includes": "Composite wellness indices, aggregate habit scores, and general health behavior patterns.",
        "excludes": "Do NOT label individual specific habits if a more specific category applies."
    },
    "Mental_health_practices": {
        "definition": "Psychological interventions, counseling techniques, and therapies aimed at mental wellness or stress management.",
        "includes": "Psychotherapy, cognitive counseling, behavioral therapy techniques, stress management, and mindfulness practices.",
        "excludes": "Do NOT label psychological disorder diagnoses—label only therapeutic practices and interventions."
    },
    "Non_physical_leisure_time_activities": {
        "definition": "Sedentary recreational pastimes, digital engagement, hobbies, and non-exertional free-time activities.",
        "includes": "Gambling, betting, screen time, video gaming, media viewing, reading, and non-physical hobbies.",
        "excludes": "Do NOT label physical exercise or active sports."
    },
    "Nutrition": {
        "definition": "Foods, beverages, macronutrients, micronutrients, dietary supplements, and dietary intake patterns.",
        "includes": "Dietary patterns, caloric intake, specific foods, beverages, vitamins, minerals, carbohydrates, and proteins.",
        "excludes": "Do NOT label pharmaceutical drugs or environmental industrial toxins."
    },
    "Physical_activity": {
        "definition": "Muscular bodily movements resulting in energy expenditure, including sports, exercise, and measured posture durations.",
        "includes": "Exercise routines, sports, active transport, walking exertion, and measured physical activity levels or sitting time.",
        "excludes": "Do NOT label background demographic settings or locations alone."
    },
    "Sleep": {
        "definition": "Physiological sleep patterns, sleep duration, quality, disturbances, and biological sleep-wake rhythms.",
        "includes": "Sleep duration, napping, sleep continuity, biological rhythms, and daytime alertness or sleepiness.",
        "excludes": "Do NOT label daytime fatigue resulting strictly from non-sleep medical conditions."
    },
    "Socioeconomic_factors": {
        "definition": "Social, economic, educational, demographic, and relationship factors influencing health and social standing.",
        "includes": "Demographics, income, education level, employment, social support, and exposure to interpersonal violence or abuse.",
        "excludes": "Do NOT label individual psychological therapies or physical exercise habits."
    }
}

# New Relabeled Category Definitions & Guidelines
LSF_CATEGORY_DEFINITIONS_RELABELED: Dict[str, Dict[str, str]] = {
    "Personal_care_products_and_cosmetic_procedures": {
        "definition": "Products, practices, and personal care items related to body grooming, cosmetics, oral hygiene, orthodontic care, and aesthetic procedures.",
        "includes": "Dental and oral care items, orthodontic appliances, skin care, cosmetics, personal care products, and aesthetic treatments.",
        "excludes": "Do NOT label medical surgical disease interventions or industrial chemical exposures."
    },
    "Substance_use": {
        "definition": "Substances, medications, addictive chemical agents, or smoking products consumed for recreational, therapeutic, or psychoactive effects.",
        "includes": "Tobacco products, cigarettes, nicotine, alcoholic beverages, illicit substances, pharmaceuticals, and smoking cessation aids.",
        "excludes": "Do NOT label standard non-drug food items or dietary nutrients."
    },
    "Environmental_exposures": {
        "definition": "External physical, chemical, or radiological agents and conditions present in workplace, ambient, or indoor air.",
        "includes": "Airborne contaminants, particulate matter (PM, SO2), coal burning, ventilation conditions, heavy metals, radiation, noise, and environmental pollutants.",
        "excludes": "Do NOT label personal voluntary habits like dietary intake or voluntary substance use."
    },
    "Lifestyle_factor": {
        "definition": "Composite or aggregate health habit indicators representing overall lifestyle behavior patterns.",
        "includes": "Composite wellness indices, aggregate habit scores, and general health behavior patterns.",
        "excludes": "Do NOT label individual specific habits if a more specific category applies."
    },
    "Mental_health_practices": {
        "definition": "Psychological interventions, counseling techniques, and therapies aimed at mental wellness, cognitive rehabilitation, or stress management.",
        "includes": "Psychotherapy, cognitive counseling, play therapy, behavioral therapy techniques, stress management, and mindfulness practices.",
        "excludes": "Do NOT label psychological disorder diagnoses—label only therapeutic practices and interventions."
    },
    "Non_physical_leisure_time_activities": {
        "definition": "Sedentary recreational pastimes, play, gambling, digital engagement, hobbies, and non-exertional free-time activities.",
        "includes": "Gambling, betting, recreational play, screen time, video gaming, media viewing, reading, and non-physical hobbies.",
        "excludes": "Do NOT label physical exercise or active sports."
    },
    "Nutrition": {
        "definition": "Foods, beverages, dietary crops, macronutrients, micronutrients, dietary supplements, and dietary intake patterns.",
        "includes": "Dietary patterns, caloric intake, specific foods, crops, propolis, beans, proteins, carbohydrates, vitamins, minerals, and dietary supplements.",
        "excludes": "Do NOT label pharmaceutical drugs or environmental industrial toxins."
    },
    "Physical_activities": {
        "definition": "Muscular bodily movements resulting in energy expenditure, including sports, exercise, strength training, active transport, and measured sedentary time.",
        "includes": "Exercise routines, strength training, sports, active transport, walking exertion, and measured physical activity levels or sitting time.",
        "excludes": "Do NOT label background demographic settings or locations alone."
    },
    "Sleep": {
        "definition": "Physiological sleep patterns, sleep duration, sleep quality, sleep deprivation (e.g. REM deprivation), disturbances, and circadian rhythms.",
        "includes": "Sleep duration, sleep deprivation, REM sleep, napping, sleep continuity, biological rhythms, and daytime alertness or sleepiness.",
        "excludes": "Do NOT label daytime fatigue resulting strictly from non-sleep medical conditions."
    },
    "Socioeconomic_factors": {
        "definition": "Social, economic, educational, demographic, occupational, and relationship factors influencing health and social standing.",
        "includes": "Demographics, schools, playgrounds, income, high/middle-income countries, education level, employment, workplaces, social support, and interpersonal factors.",
        "excludes": "Do NOT label individual psychological therapies or physical exercise habits."
    }
}

# Default backwards-compatible dictionary
LSF_CATEGORY_DEFINITIONS = LSF_CATEGORY_DEFINITIONS_LEGACY
LSF_CATEGORY_DEFINITIONS_AND_GUIDELINES = LSF_CATEGORY_DEFINITIONS_LEGACY


def get_category_definitions(use_new_labels: bool = False) -> Dict[str, Dict[str, str]]:
    """Returns category definitions matching the selected ontology schema."""
    return LSF_CATEGORY_DEFINITIONS_RELABELED if use_new_labels else LSF_CATEGORY_DEFINITIONS_LEGACY


def format_d_and_g_text(use_new_labels: bool = False) -> str:
    """Format category Definitions & Guidelines into markdown text for prompt injection."""
    definitions = get_category_definitions(use_new_labels)
    text = ""
    for category, dng in definitions.items():
        text += f"- **{category}**:\n"
        text += f"  - *Definition*: {dng['definition']}\n"
        text += f"  - *Includes*: {dng['includes']}\n"
        text += f"  - *Excludes*: {dng['excludes']}\n\n"
        
    text += "- **O**: The mention is NOT a valid lifestyle entity or does not belong to any category above (Out of categories / non-entity background)."
    return text


def build_lsf_linkner_prompt(context_text: str, candidate_span: str, use_new_labels: bool = False) -> str:
    """
    Constructs a domain-specific LinkNER Multiple-Choice Prompt for the LSF dataset.
    """
    category_list_text = format_d_and_g_text(use_new_labels=use_new_labels)
    sample_cat = "Physical_activities" if use_new_labels else "Physical_activity"

    prompt = f"""You are an expert biomedical NLP entity classifier specializing in Lifestyle Factors (LSF).

Context Sentence/Abstract:
"{context_text}"

Candidate Mention to Classify:
"{candidate_span}"

Candidate Categories, Enriched Definitions & Guidelines:
{category_list_text}

Instructions & Disambiguation Rules:
1. Examine the candidate mention "{candidate_span}" in the context above.
2. Contextual Disambiguation: Compare the candidate mention against the Definitions, Includes, and Excludes guidelines above.
3. Affirmative Filtering (Mandatory Exclusions -> "O"):
   - Internal biomarkers/metabolites (e.g. blood glucose, cholesterol) -> "O".
   - Demographics (e.g. elderly, adolescents) -> "O" (unless evaluated strictly under Socioeconomic_factors).
   - Standalone adjectives (e.g. nutritional, occupational) -> "O".
   - Non-lifestyle medical/surgical procedures (only psychotherapy & cosmetic procedures qualify as LSFs) -> "O".
   - Study group names / acronyms (e.g. "SMOKE study") -> "O".
4. Special Handling Checks:
   - Metrics, scores, and indices that quantify an LSF (e.g. "sleep quality score", "physical activity index") MUST be labeled with their corresponding LSF category.
   - LSF abbreviations (e.g. "PA" for Physical_activities) in lifestyle context MUST be labeled with their corresponding category.
5. Specific Category Priority: Always prefer specific action/habit categories (Physical_activities/Physical_activity, Nutrition, Sleep, Substance_use/Drugs) over the generic aggregate Lifestyle_factor whenever specific bodily exertion, dietary intake, or substance use is present.
6. Select the SINGLE most accurate category from the list above (or "O" if not a valid entity).
7. Respond ONLY with the exact category name (e.g., "{sample_cat}" or "O"). Do NOT add preamble or explanations.

Classification:"""

    return prompt


FEW_SHOT_EXEMPLARS_TEXT_LEGACY = """Exemplar Demonstrations:
Example Context 1: "We evaluated the effect of structured aerobic exercise on cardiovascular health among high school seniors."
Candidate Mentions:
1. "aerobic exercise" -> 1. Physical_activity
2. "high school seniors" -> 2. Socioeconomic_factors

Example Context 2: "Substance abuse treatment included cognitive therapy and nicotine replacement."
Candidate Mentions:
1. "cognitive therapy" -> 1. Mental_health_practices
2. "nicotine replacement" -> 2. Drugs

Example Context 3 (Compound Mentions): "Study examined the relationship between religion and alcohol consumption."
Candidate Mentions:
1. "religion and alcohol consumption" -> 1a. "religion" -> Socioeconomic_factors
1b. "alcohol consumption" -> Nutrition
"""

FEW_SHOT_EXEMPLARS_TEXT_RELABELED = """Exemplar Demonstrations:
Example Context 1: "We evaluated the effect of structured aerobic exercise on cardiovascular health among high school seniors."
Candidate Mentions:
1. "aerobic exercise" -> 1. Physical_activities
2. "high school seniors" -> 2. Socioeconomic_factors

Example Context 2: "Substance abuse treatment included cognitive therapy and nicotine replacement."
Candidate Mentions:
1. "cognitive therapy" -> 1. Mental_health_practices
2. "nicotine replacement" -> 2. Substance_use

Example Context 3 (Compound Mentions): "Study examined the relationship between religion and alcohol consumption."
Candidate Mentions:
1. "religion and alcohol consumption" -> 1a. "religion" -> Socioeconomic_factors
1b. "alcohol consumption" -> Nutrition
"""

FEW_SHOT_EXEMPLARS_TEXT = FEW_SHOT_EXEMPLARS_TEXT_LEGACY


def build_lsf_linkner_batch_prompt(
    context_text: str,
    candidate_spans: list,
    enable_few_shot: bool = True,
    use_new_labels: bool = False
) -> str:
    """
    Constructs a domain-specific LinkNER Multiple-Choice Prompt for the LSF dataset,
    handling multiple candidate mentions in a single prompt.
    """
    category_list_text = format_d_and_g_text(use_new_labels=use_new_labels)

    spans_list_text = ""
    for i, item in enumerate(candidate_spans, 1):
        if isinstance(item, dict):
            span_text = item.get("span_text", str(item))
            s_label = item.get("spanner_label", "O")
            if s_label != "O":
                spans_list_text += f'{i}. "{span_text}" (Primary Model Suggestion: {s_label})\n'
            else:
                spans_list_text += f'{i}. "{span_text}"\n'
        else:
            spans_list_text += f'{i}. "{item}"\n'

    few_shot_template = FEW_SHOT_EXEMPLARS_TEXT_RELABELED if use_new_labels else FEW_SHOT_EXEMPLARS_TEXT_LEGACY
    few_shot_section = (few_shot_template + "\n") if enable_few_shot else ""
    sample_cat = "Physical_activities" if use_new_labels else "Physical_activity"

    prompt = f"""You are an expert biomedical NLP entity classifier specializing in Lifestyle Factors (LSF).

Context Sentence/Abstract:
"{context_text}"

Candidate Mentions to Classify:
{spans_list_text}
Candidate Categories, Enriched Definitions & Guidelines:
{category_list_text}

{few_shot_section}Instructions & Disambiguation Rules:
1. Contextual Disambiguation: Examine each candidate mention in context. Utilize the exact Definitions, Includes, and Excludes guidelines for each category above.
2. Mandatory Compound Mention Splitting Rule: If a candidate mention contains multiple distinct items or words joined by connectors or conjunctions ('and', 'or', 'as', ...), ALWAYS split the candidate mention into sub-mentions and evaluate each constituent item on its own line (e.g. 1a. "food" -> Nutrition, 1b. "water" -> Nutrition). Do NOT output a single "O" for compound mentions if individual parts represent valid entities.
3. Affirmative Filtering (Mandatory Exclusions -> "O"):
   - Internal biomarkers/metabolites (e.g. blood glucose, cholesterol, insulin) -> "O".
   - Demographics (e.g. elderly, adolescents) -> "O" (unless the primary study focus is socioeconomic status/demographic disparity).
   - Standalone adjectives (e.g. nutritional, occupational) -> "O".
   - Non-lifestyle medical procedures (only psychotherapy & cosmetic procedures qualify as LSFs; hospital/clinical surgeries -> "O").
   - Non-behavioral study names / acronyms (e.g. "SMOKE study group") -> "O".
4. Special Handling Checks:
   - Metrics, scores, and indices that quantify an LSF (e.g. "sleep quality score", "physical activity index") MUST be labeled with their corresponding LSF category.
   - LSF abbreviations (e.g. "PA" for Physical_activities) in lifestyle context MUST be labeled.
5. Specific Category Priority Rule: Always prefer specific action/habit categories (Physical_activities/Physical_activity, Nutrition, Sleep, Substance_use/Drugs) over the generic aggregate umbrella category Lifestyle_factor whenever physical exertion, active living, bodily movement, dietary intake, or substance use is present.
6. Human Dietary & Hydration Intake Rule: Classify phrases evaluating human food ingestion, beverage consumption, dietary hydration, or caloric intake under Nutrition. If a chemical or liquid compound is mentioned purely as an abstract laboratory assay substrate or chemical reagent without dietary consumption context, assign "O".
7. Utilize Model Suggestions & Preserve Valid Entities: Consider the Primary Model Suggestion provided for each candidate. If the candidate mention represents a valid lifestyle entity (e.g. exercise, sports, foods, nutrients, sleep, environmental exposures), assign its matching category. Do NOT output "O" for valid entities.
8. Select the SINGLE most accurate category (or split sub-mentions) from the list above for each mention (or "O" if out of categories).
9. Respond ONLY with a line-by-line list matching the index of the mention to the exact category name (or sub-mention format) (e.g., "1. {sample_cat}", "1a. \\"religion\\" -> Socioeconomic_factors", or "1. O"). Do NOT add preamble or explanations.

Classification:"""

    return prompt

