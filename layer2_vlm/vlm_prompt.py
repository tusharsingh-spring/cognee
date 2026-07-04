"""VLM prompt builder — constructs prompts that tell the VLM what was already detected."""

from typing import Optional


def build_person_vlm_prompt(perception_context: str, task: str = "dense") -> str:
    if task == "dense":
        return f"""<MORE_DETAILED_CAPTION>

Perception models already detected: {perception_context}

You are the visual context layer. Describe this person comprehensively.
Focus on what the structured models CANNOT capture:
- Emotional state and mood
- Clothing: colors, patterns, fabric type, style
- Exact body posture and stance
- Objects they are holding or interacting with
- Facial expression (happy, worried, neutral, angry)
- Any unusual behavior or visual anomalies
- Their apparent intent based on body language

Provide a detailed 4-6 sentence description."""

    elif task == "scene":
        return f"""<DETAILED_CAPTION>
Scene context from models: {perception_context}
Describe the scene around this person. What's in the background?
What is the environment like? Any notable elements?"""

    elif task == "od":
        return f"""<OD>
Detect all objects in this image. Already known: {perception_context}
List any additional objects the models may have missed."""

    return f"""<DETAILED_CAPTION>
{perception_context}
Describe this image."""


def build_full_scene_vlm_prompt(perception_context: str) -> str:
    return f"""<MORE_DETAILED_CAPTION>

Overall scene context from perception: {perception_context}

You are the visual context layer. Describe the COMPLETE scene:
- How many people? What are they doing relative to each other?
- What is the environment? (indoor/outdoor, room type, lighting)
- Any unusual activity or anomalies?
- Overall mood and atmosphere
- Time of day indicators
- Any safety concerns visible

Provide a comprehensive 4-6 sentence scene description."""


def parse_vlm_response(response: str) -> dict:
    """Try to parse VLM output as structured JSON. Falls back to raw text."""
    import json

    try:
        if "{" in response and "}" in response:
            start = response.index("{")
            end = response.rindex("}") + 1
            json_str = response[start:end]
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass

    return {
        "visual_context": response,
        "emotional_cues": "",
        "additional_objects": "",
        "scene_description": response,
    }
