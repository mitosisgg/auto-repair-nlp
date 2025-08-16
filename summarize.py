import requests
import json
import argparse
import spacy
from typing import Dict, Any
import re
from utils import preprocess_texts, get_stopwords, safe_text

# Lazy-loaded spaCy model
_NLP = None

def get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            # If spaCy model isn't available, leave as None (preprocess_texts will fallback)
            _NLP = None
    return _NLP

def get_llm_api_url(host: str = 'localhost', port: int = 12434) -> str:
    """Construct the LLM API URL from host and port"""
    return f"http://{host}:{port}/engines/llama.cpp/v1/chat/completions"

def summarize_descriptions(issue_description: str,
                           repair_description: str,
                           model_name: str = 'gemma3',
                           api_url: str = None,
                           llm_host: str = 'localhost',
                           llm_port: int = 12434) -> Dict[str, Any]:
    # If api_url not provided, construct it from host and port
    if api_url is None:
        api_url = get_llm_api_url(llm_host, llm_port)

    """Summarize and classify a car repair record using a local LLM."""

    # Preprocess inputs using utils.preprocess_texts with spaCy if available
    stopwords, keepwords = get_stopwords()
    nlp = get_nlp()
    preprocessed_issue = preprocess_texts([safe_text(issue_description)], nlp=nlp, stopwords=stopwords, keepwords=keepwords)
    preprocessed_repair = preprocess_texts([safe_text(repair_description)], nlp=nlp, stopwords=stopwords, keepwords=keepwords)


    system_prompt = """
        You are a strict JSON generator. You return exactly one JSON object and nothing else:
        - No prose, no explanations, no markdown, no code fences.
        - Use double quotes for all keys and string values.
        - All enum values must match only one of the allowed options exactly.
        - Do not omit opening or closing brackets in your JSON response.
    """

    user_prompt = f"""
        INPUT
        issue_description: {preprocessed_issue}
        repair_description: {preprocessed_repair}

        TASK
        1. Summarize issue_description in 1-2 sentences → "ai04c__complaint__issue_description"
        2. Summarize repair_description in 1-2 sentences → "ai04__repair_details"
        3. Classify the note using ONLY the following allowed options:

        - "ai04g__issue_presentation": ["recall/campaign","customer complaint","regular maintenance","dealer discovered"]
        - "ai04h__issue_type": ["Brakes","Body/Chassis","Turn Signals","Back-Up Lights","SRS System","Airbag","Neck-Pro","Collision Prevent Assist","Safety Systems (DNU)","MPI","Oil Change","PDI","Seat Belt / Seat Belt Tensioner","Routine Maintenance","Miscellaneous - Non-Conformity","Battery Drainage / No Start","Climate Control System","Driver Assist Features","Drivetrain","Eco Start/Stop","Electric Drivetrain","Engine / Trans Noise","Engine Function - Other","Fluid Leaks","Fuel Consumption  / Range","Fuel System","Infotainment / Telematics","Intake/ Exhaust","Interior Button / Switches","Interior Trim / Fit Finish","Key / Keyless-Go","Misbuild / Missing Options","Non-Technical Issue","Odor","Paint","Roof","Seats","Shifting","Stalling / Idling","Steering","Suspension","Thermal Event","Transmission Function - Other","Water Intrusion","Wheel / Tire","Windows / Windshield"]
        - "ai04m__repair_costs_handling": ["Customer Pay","OEM Warranty","OEM Goodwill","Dealer Goodwill","Prepaid Maintenance"]
        - "ai04s__does_repair_fall_under_warranty": ["Initial Warranty","CPO","Extended Warranty"]
        - "ai04i__issue_verified": ["Yes","No"]
        - "ai04r__oem_engineering_services_involved": ["Yes","No"]
        - "ai04j__repair_performed": ["Yes","No"]
        - "ai04k___of_repairs_performed_for_this_issue": [0,1,2,3,4,5,6,7,8,9,10]
        - "ai04n__not_repaired_reason": ["Customer Declined Service","Unable to Replicate / Verify","Part Availability","Unable to Repair / No Technical Resolution","Repair Pending"]
        - "ai04l__is_this_issue_the_primary_issue_driving_the_days_down": ["Yes","No"]
        - "ai04o__days_out_reason": ["Part Availability","Software Availability","Complex Diagnosis","Complex Repair","Dealer Process","Other"]
        - "ai04q__outside_influences": ["Client declined repairs at this time, pick up vehicle for work purposes.","Regular maintenance","Waiting for label","Awaiting label issue a","Routine maintenance","Unable to duplicate","Campaign/recall","Unable to verify","Complex repair","Multiple complaints and repairs","Unable to replicate","Part availability","Sold before work performed","General repair","Windshield replacement","Sublet repair to other shop","Aftermarket parts","Unable to duplicate to diagnose","Tire replacement","RO lists multiple issues stated by the client","Insurance claim - car accident","Checking all shocks for fluid leaks","Customer could not wait for vehicle to be repaired","Recall b-pillar","Complex diagnosis","All icons starting flashing car lost power would not accelerate","PDI","Multiple complaints","RO created","Federal holiday","Dealer process","Software availability","Body shop referral","Holiday weekend", "N/A"]

        OUTPUT
        Return JSON with exactly this shape:

        {{
        "ai04c__complaint__issue_description": "<string>",
        "ai04__repair_details": "<string>",
        "ai04g__issue_presentation": "<enum string>",
        "ai04h__issue_type": "<enum string>",
        "ai04m__repair_costs_handling": "<enum string>",
        "ai04s__does_repair_fall_under_warranty": "<enum string>",
        "ai04i__issue_verified": "<enum string>",
        "ai04r__oem_engineering_services_involved": "<enum string>",
        "ai04j__repair_performed": "<enum string>",
        "ai04k___of_repairs_performed_for_this_issue": <integer>,
        "ai04n__not_repaired_reason": "<enum string>",
        "ai04l__is_this_issue_the_primary_issue_driving_the_days_down": "<enum string>",
        "ai04o__days_out_reason": "<enum string>",
        "ai04q__outside_influences": "<enum string>"
        }}
"""

    user_prompt2 = f"""
        You are a JSON generator. Return exactly one JSON object and nothing else:
        - No prose, explanations, or markdown.
        - No code fences.
        - All keys and string values must use double quotes.
        - Summarize clearly and concisely.

        INPUT:
        issue_description: {preprocessed_issue}
        repair_description: {preprocessed_repair}

        TASK:
        1. Summarize issue_description in 1–2 sentences, removing part numbers, prices, or internal codes. Output as "ai04c__complaint__issue_description".
        2. Summarize repair_description in 1–2 sentences, removing part numbers, prices, or internal codes. Output as "ai04__repair_details".

        OUTPUT:
        Return a JSON object with exactly this shape:

        {{
        "ai04c__complaint__issue_description": "<string>",
        "ai04__repair_details": "<string>"
        }}
    """

    headers = {'Content-Type': 'application/json'}
    payload = {
        'model': f'ai/{model_name}',
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt2}
        ],
        "response_format": { "type": "json_object" }
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()
        data = response.json()
        content = data['choices'][0]['message']['content']
        
        # Clean up any remaining non-JSON content
        content = content.strip()
        
        # Parse the JSON
        result = json.loads(content)
        return result
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response:\n {content}")
        return {"error": "Failed to parse model response"}
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        return {"error": str(e)}
