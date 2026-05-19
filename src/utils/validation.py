import json

def attempt_json_recovery(truncated_json_str):
    """Attempts to close a truncated JSON string for partial extraction."""
    temp_str = truncated_json_str.strip()
    for _ in range(5):
        try:
            return json.loads(temp_str)
        except json.JSONDecodeError:
            if temp_str.endswith('"'): temp_str += ' }'
            elif temp_str.endswith(','): temp_str = temp_str[:-1] + ' }'
            else: temp_str += ' }'
    return {"requires_human_review": True, "error": "JSON Truncated"}

def ensure_structure(data):
    """Ensures the extracted JSON has all required top-level keys."""
    defaults = {
        "document_details": {},
        "vendor_details": {},
        "client_details": {},
        "line_items": [],
        "financials": {},
        "requires_human_review": False,
        "validation_errors": []
    }
    for key, value in defaults.items():
        if key not in data:
            data[key] = value
    return data

def validate_extraction(data):
    """Validates the extracted JSON data for errors and mathematical consistency."""
    data = ensure_structure(data)
    issues = []
    
    financials = data.get("financials", {})
    subtotal = financials.get("subtotal") or 0.0
    tax = financials.get("tax_amount") or 0.0
    total = financials.get("total_amount") or 0.0
    
    # Math Check
    expected_total = subtotal + tax
    if abs(expected_total - total) > 0.02:
        issues.append(f"Math mismatch: Subtotal({subtotal}) + Tax({tax}) != Total({total})")
    
    # Critical Fields Check - Removed to be lenient for continuation pages
    # if not data.get("vendor_details", {}).get("company_name"):
    #     issues.append("Missing Vendor Name")
    
    if issues:
        data["requires_human_review"] = True
        data["validation_errors"] = issues
    else:
        # Preserve existing flag if recovery failed previously
        data["requires_human_review"] = data.get("requires_human_review", False)
        
    return data
