import os
import csv
import uuid
from datetime import datetime

class CSVDatabase:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.invoices_csv = os.path.join(self.data_dir, "invoices.csv")
        self.line_items_csv = os.path.join(self.data_dir, "line_items.csv")
        os.makedirs(self.data_dir, exist_ok=True)
        self._initialize_csv_files()

    def _initialize_csv_files(self):
        """Creates the CSV files with headers if they don't exist."""
        
        # Invoices (Headers & Totals)
        if not os.path.exists(self.invoices_csv):
            with open(self.invoices_csv, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "invoice_id", "processed_at",
                    "doc_type", "invoice_number", "invoice_date", "due_date",
                    "vendor_company", "vendor_person", "vendor_address",
                    "client_company", "client_person", "client_address",
                    "subtotal", "tax_amount", "total_amount",
                    "requires_human_review", "validation_errors"
                ])
                
        # Line Items (Relational Table)
        if not os.path.exists(self.line_items_csv):
            with open(self.line_items_csv, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "item_id", "invoice_id",
                    "description", "quantity", "unit_price", "line_total"
                ])

    def save_extraction(self, diagnostic_pages):
        """
        Saves the unmerged 'Glass Pipeline' diagnostic pages into a normalized, 
        relational CSV structure.
        """
        if not diagnostic_pages:
            return None
            
        invoice_id = str(uuid.uuid4())
        processed_at = datetime.now().isoformat()
        
        # Merge data strictly for database storage (finding first non-null values)
        merged_sd = {
            "document_details": {}, "vendor_details": {}, "client_details": {},
            "line_items": [], "financials": {}, "validation_errors": [],
            "requires_human_review": False
        }
        
        for page in diagnostic_pages:
            sd = page.get("stage_5_llm_json", {})
            if sd.get("requires_human_review"):
                merged_sd["requires_human_review"] = True
            if sd.get("validation_errors"):
                merged_sd["validation_errors"].extend(sd["validation_errors"])
                
            for section in ["document_details", "vendor_details", "client_details", "financials"]:
                s_data = sd.get(section, {})
                for k, v in s_data.items():
                    # Keep first non-null/non-empty for header info, overwrite for financials
                    if v and (not merged_sd[section].get(k) or section == "financials"):
                        merged_sd[section][k] = v
                        
            if sd.get("line_items"):
                merged_sd["line_items"].extend(sd["line_items"])

        # 1. Save to invoices.csv
        dd = merged_sd["document_details"]
        vd = merged_sd["vendor_details"]
        cd = merged_sd["client_details"]
        fin = merged_sd["financials"]
        
        with open(self.invoices_csv, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                invoice_id, processed_at,
                dd.get("document_type", ""), dd.get("invoice_number", ""), dd.get("invoice_date", ""), dd.get("due_date", ""),
                vd.get("company_name", ""), vd.get("person_name", ""), vd.get("address", ""),
                cd.get("company_name", ""), cd.get("person_name", ""), cd.get("address", ""),
                fin.get("subtotal", 0.0), fin.get("tax_amount", 0.0), fin.get("total_amount", 0.0),
                merged_sd["requires_human_review"], "; ".join(merged_sd["validation_errors"])
            ])

        # 2. Save to line_items.csv
        with open(self.line_items_csv, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for item in merged_sd["line_items"]:
                item_id = str(uuid.uuid4())
                writer.writerow([
                    item_id, invoice_id,
                    item.get("description", ""),
                    item.get("quantity", 0.0),
                    item.get("unit_price", 0.0),
                    item.get("line_total", 0.0)
                ])
                
        return invoice_id