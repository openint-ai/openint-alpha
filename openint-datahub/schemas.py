"""
Schema definitions for openInt testdata datasets.
These schemas define the structure of each dataset for DataHub metadata ingestion.
"""

from typing import List, Dict, Any

# Import DataHub classes only when needed (in ingest_metadata.py)
# This allows schemas.py to be imported without DataHub SDK installed


def get_dataset_schemas() -> Dict[str, Dict[str, Any]]:
    """
    Returns schema definitions for all openInt datasets.
    
    Returns:
        Dictionary mapping dataset names to their schema definitions
    """
    schemas = {
        # Dimension Tables
        "customers": {
            "description": "Customer dimension table containing customer profile information",
            "category": "dimension",
            "fields": [
                {"name": "customer_id", "type": "STRING", "description": "Unique customer identifier (CUST format)"},
                {"name": "ssn", "type": "STRING", "description": "Social Security Number"},
                {"name": "first_name", "type": "STRING", "description": "Customer first name"},
                {"name": "last_name", "type": "STRING", "description": "Customer last name"},
                {"name": "email", "type": "STRING", "description": "Customer email address"},
                {"name": "phone", "type": "STRING", "description": "Customer phone number"},
                {"name": "date_of_birth", "type": "DATE", "description": "Customer date of birth"},
                {"name": "account_opened_date", "type": "DATE", "description": "Date when account was opened"},
                {"name": "street_address", "type": "STRING", "description": "Street address"},
                {"name": "city", "type": "STRING", "description": "City name"},
                {"name": "state_code", "type": "STRING", "description": "US state code (2-letter)"},
                {"name": "zip_code", "type": "STRING", "description": "ZIP code"},
                {"name": "country_code", "type": "STRING", "description": "Country code (2-letter ISO)"},
                {"name": "customer_type", "type": "STRING", "description": "Customer type (Individual/Business)"},
                {"name": "account_status", "type": "STRING", "description": "Account status (Active/Closed)"},
                {"name": "credit_score", "type": "NUMBER", "description": "Customer credit score"},
                {"name": "created_at", "type": "DATETIME", "description": "Record creation timestamp"},
                {"name": "updated_at", "type": "DATETIME", "description": "Record last update timestamp"},
            ]
        },
        
        # Fact Tables - Transactions
        "ach_transactions": {
            "description": "ACH (Automated Clearing House) transaction fact table",
            "category": "fact",
            "fields": [
                {"name": "transaction_id", "type": "STRING", "description": "Unique transaction identifier"},
                {"name": "customer_id", "type": "STRING", "description": "Foreign key to customers table"},
                {"name": "transaction_date", "type": "DATE", "description": "Transaction date"},
                {"name": "transaction_datetime", "type": "DATETIME", "description": "Transaction timestamp"},
                {"name": "transaction_type", "type": "STRING", "description": "Transaction type (Credit/Debit)"},
                {"name": "amount", "type": "NUMBER", "description": "Transaction amount"},
                {"name": "currency", "type": "STRING", "description": "Currency code (USD)"},
                {"name": "ach_code", "type": "STRING", "description": "ACH transaction code (POP, CCD, etc.)"},
                {"name": "routing_number", "type": "STRING", "description": "Bank routing number"},
                {"name": "account_number", "type": "STRING", "description": "Account number"},
                {"name": "description", "type": "STRING", "description": "Transaction description"},
                {"name": "status", "type": "STRING", "description": "Transaction status"},
                {"name": "created_at", "type": "DATETIME", "description": "Record creation timestamp"},
            ]
        },
        
        "wire_transactions": {
            "description": "Wire transfer transaction fact table",
            "category": "fact",
            "fields": [
                {"name": "transaction_id", "type": "STRING", "description": "Unique transaction identifier"},
                {"name": "customer_id", "type": "STRING", "description": "Foreign key to customers table"},
                {"name": "transaction_date", "type": "DATE", "description": "Transaction date"},
                {"name": "transaction_datetime", "type": "DATETIME", "description": "Transaction timestamp"},
                {"name": "wire_type", "type": "STRING", "description": "Wire type (Domestic/International)"},
                {"name": "amount", "type": "NUMBER", "description": "Transaction amount"},
                {"name": "currency", "type": "STRING", "description": "Currency code"},
                {"name": "sender_routing", "type": "STRING", "description": "Sender routing number"},
                {"name": "beneficiary_routing", "type": "STRING", "description": "Beneficiary routing number"},
                {"name": "beneficiary_account", "type": "STRING", "description": "Beneficiary account number"},
                {"name": "beneficiary_name", "type": "STRING", "description": "Beneficiary name"},
                {"name": "beneficiary_country", "type": "STRING", "description": "Beneficiary country code"},
                {"name": "beneficiary_bank_swift", "type": "STRING", "description": "SWIFT code for international wires"},
                {"name": "fee", "type": "NUMBER", "description": "Wire transfer fee"},
                {"name": "description", "type": "STRING", "description": "Transaction description"},
                {"name": "status", "type": "STRING", "description": "Transaction status"},
                {"name": "created_at", "type": "DATETIME", "description": "Record creation timestamp"},
            ]
        },
        
        "credit_transactions": {
            "description": "Credit card transaction fact table",
            "category": "fact",
            "fields": [
                {"name": "transaction_id", "type": "STRING", "description": "Unique transaction identifier"},
                {"name": "customer_id", "type": "STRING", "description": "Foreign key to customers table"},
                {"name": "transaction_date", "type": "DATE", "description": "Transaction date"},
                {"name": "transaction_datetime", "type": "DATETIME", "description": "Transaction timestamp"},
                {"name": "card_type", "type": "STRING", "description": "Card type (Visa, Mastercard, etc.)"},
                {"name": "card_number_last4", "type": "STRING", "description": "Last 4 digits of card number"},
                {"name": "amount", "type": "NUMBER", "description": "Transaction amount"},
                {"name": "currency", "type": "STRING", "description": "Currency code"},
                {"name": "merchant_name", "type": "STRING", "description": "Merchant name"},
                {"name": "merchant_category", "type": "STRING", "description": "Merchant category code"},
                {"name": "merchant_city", "type": "STRING", "description": "Merchant city"},
                {"name": "merchant_state", "type": "STRING", "description": "Merchant state code"},
                {"name": "merchant_country", "type": "STRING", "description": "Merchant country code"},
                {"name": "authorization_code", "type": "STRING", "description": "Authorization code"},
                {"name": "description", "type": "STRING", "description": "Transaction description"},
                {"name": "status", "type": "STRING", "description": "Transaction status"},
                {"name": "created_at", "type": "DATETIME", "description": "Record creation timestamp"},
            ]
        },
        
        "debit_transactions": {
            "description": "Debit card transaction fact table",
            "category": "fact",
            "fields": [
                {"name": "transaction_id", "type": "STRING", "description": "Unique transaction identifier"},
                {"name": "customer_id", "type": "STRING", "description": "Foreign key to customers table"},
                {"name": "transaction_date", "type": "DATE", "description": "Transaction date"},
                {"name": "transaction_datetime", "type": "DATETIME", "description": "Transaction timestamp"},
                {"name": "transaction_type", "type": "STRING", "description": "Transaction type"},
                {"name": "amount", "type": "NUMBER", "description": "Transaction amount"},
                {"name": "currency", "type": "STRING", "description": "Currency code"},
                {"name": "merchant_name", "type": "STRING", "description": "Merchant name"},
                {"name": "merchant_category", "type": "STRING", "description": "Merchant category"},
                {"name": "merchant_location", "type": "STRING", "description": "Merchant location"},
                {"name": "description", "type": "STRING", "description": "Transaction description"},
                {"name": "status", "type": "STRING", "description": "Transaction status"},
                {"name": "created_at", "type": "DATETIME", "description": "Record creation timestamp"},
            ]
        },
        
        "check_transactions": {
            "description": "Check transaction fact table",
            "category": "fact",
            "fields": [
                {"name": "transaction_id", "type": "STRING", "description": "Unique transaction identifier"},
                {"name": "customer_id", "type": "STRING", "description": "Foreign key to customers table"},
                {"name": "transaction_date", "type": "DATE", "description": "Transaction date"},
                {"name": "transaction_datetime", "type": "DATETIME", "description": "Transaction timestamp"},
                {"name": "check_number", "type": "STRING", "description": "Check number"},
                {"name": "amount", "type": "NUMBER", "description": "Transaction amount"},
                {"name": "currency", "type": "STRING", "description": "Currency code"},
                {"name": "payee_name", "type": "STRING", "description": "Payee name"},
                {"name": "memo", "type": "STRING", "description": "Check memo"},
                {"name": "description", "type": "STRING", "description": "Transaction description"},
                {"name": "status", "type": "STRING", "description": "Transaction status"},
                {"name": "created_at", "type": "DATETIME", "description": "Record creation timestamp"},
            ]
        },
        
        "disputes": {
            "description": "Transaction dispute fact table",
            "category": "fact",
            "fields": [
                {"name": "dispute_id", "type": "STRING", "description": "Unique dispute identifier"},
                {"name": "transaction_type", "type": "STRING", "description": "Type of transaction (ach, wire, credit, etc.)"},
                {"name": "transaction_id", "type": "STRING", "description": "Foreign key to transaction table"},
                {"name": "customer_id", "type": "STRING", "description": "Foreign key to customers table"},
                {"name": "dispute_date", "type": "DATE", "description": "Date dispute was filed"},
                {"name": "dispute_reason", "type": "STRING", "description": "Reason for dispute"},
                {"name": "dispute_status", "type": "STRING", "description": "Dispute status"},
                {"name": "amount_disputed", "type": "NUMBER", "description": "Amount being disputed"},
                {"name": "currency", "type": "STRING", "description": "Currency code"},
                {"name": "description", "type": "STRING", "description": "Dispute description"},
                {"name": "created_at", "type": "DATETIME", "description": "Record creation timestamp"},
            ]
        },
        
        # Static Dimension Tables
        "country_codes": {
            "description": "Country codes reference table",
            "category": "static",
            "fields": [
                {"name": "country_code", "type": "STRING", "description": "2-letter country code"},
                {"name": "country_name", "type": "STRING", "description": "Country name"},
                {"name": "iso_code", "type": "STRING", "description": "ISO 3-letter country code"},
                {"name": "region", "type": "STRING", "description": "Geographic region"},
            ]
        },
        
        "state_codes": {
            "description": "US state codes reference table",
            "category": "static",
            "fields": [
                {"name": "state_code", "type": "STRING", "description": "2-letter state code"},
                {"name": "state_name", "type": "STRING", "description": "State name"},
                {"name": "region", "type": "STRING", "description": "US region (Northeast, South, Midwest, West)"},
            ]
        },
        
        "zip_codes": {
            "description": "ZIP code reference table with geographic information",
            "category": "static",
            "fields": [
                {"name": "zip_code", "type": "STRING", "description": "ZIP code"},
                {"name": "city", "type": "STRING", "description": "City name"},
                {"name": "state_code", "type": "STRING", "description": "State code"},
                {"name": "latitude", "type": "NUMBER", "description": "Latitude coordinate"},
                {"name": "longitude", "type": "NUMBER", "description": "Longitude coordinate"},
                {"name": "timezone", "type": "STRING", "description": "Timezone"},
            ]
        },
    }
    
    return schemas
