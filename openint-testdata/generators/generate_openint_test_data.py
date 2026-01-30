"""
openInt Test Data Generator
Generates realistic openInt test data including:
- Customer dimension table (default 10K records; override with --num-customers)
- Transaction fact tables (ACH, Wire, Credit, Debit, Check) for those customers only
- Static dimension tables (country codes, state codes, zip codes)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from pathlib import Path

# Initialize Faker
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# Create testdata directory structure
BASE_DIR = Path("testdata")
BASE_DIR.mkdir(exist_ok=True)

# Subdirectories
DIMENSIONS_DIR = BASE_DIR / "dimensions"
FACTS_DIR = BASE_DIR / "facts"
STATIC_DIR = BASE_DIR / "static"

for dir_path in [DIMENSIONS_DIR, FACTS_DIR, STATIC_DIR]:
    dir_path.mkdir(exist_ok=True)


# Batch size for generation loops: 10K–50K based on system; override with GENERATE_BATCH_SIZE
def _default_generate_batch_size() -> int:
    try:
        env_val = os.environ.get("GENERATE_BATCH_SIZE")
        if env_val is not None:
            return max(10_000, min(50_000, int(env_val)))
    except (TypeError, ValueError):
        pass
    cpu = (os.cpu_count() or 4)
    if cpu >= 8:
        return 50_000
    if cpu >= 4:
        return 25_000
    return 10_000


def generate_country_codes():
    """Generate country codes dimension table"""
    countries = [
        {"country_code": "US", "country_name": "United States", "iso_code": "USA", "region": "North America"},
        {"country_code": "CA", "country_name": "Canada", "iso_code": "CAN", "region": "North America"},
        {"country_code": "MX", "country_name": "Mexico", "iso_code": "MEX", "region": "North America"},
        {"country_code": "GB", "country_name": "United Kingdom", "iso_code": "GBR", "region": "Europe"},
        {"country_code": "DE", "country_name": "Germany", "iso_code": "DEU", "region": "Europe"},
        {"country_code": "FR", "country_name": "France", "iso_code": "FRA", "region": "Europe"},
        {"country_code": "IT", "country_name": "Italy", "iso_code": "ITA", "region": "Europe"},
        {"country_code": "ES", "country_name": "Spain", "iso_code": "ESP", "region": "Europe"},
        {"country_code": "NL", "country_name": "Netherlands", "iso_code": "NLD", "region": "Europe"},
        {"country_code": "BE", "country_name": "Belgium", "iso_code": "BEL", "region": "Europe"},
        {"country_code": "CH", "country_name": "Switzerland", "iso_code": "CHE", "region": "Europe"},
        {"country_code": "AU", "country_name": "Australia", "iso_code": "AUS", "region": "Oceania"},
        {"country_code": "NZ", "country_name": "New Zealand", "iso_code": "NZL", "region": "Oceania"},
        {"country_code": "JP", "country_name": "Japan", "iso_code": "JPN", "region": "Asia"},
        {"country_code": "CN", "country_name": "China", "iso_code": "CHN", "region": "Asia"},
        {"country_code": "IN", "country_name": "India", "iso_code": "IND", "region": "Asia"},
        {"country_code": "KR", "country_name": "South Korea", "iso_code": "KOR", "region": "Asia"},
        {"country_code": "SG", "country_name": "Singapore", "iso_code": "SGP", "region": "Asia"},
        {"country_code": "BR", "country_name": "Brazil", "iso_code": "BRA", "region": "South America"},
        {"country_code": "AR", "country_name": "Argentina", "iso_code": "ARG", "region": "South America"},
    ]
    df = pd.DataFrame(countries)
    df.to_csv(STATIC_DIR / "country_codes.csv", index=False)
    print(f"✅ Generated country_codes.csv with {len(df)} records")
    return df


def generate_state_codes():
    """Generate US state codes dimension table"""
    us_states = [
        {"state_code": "AL", "state_name": "Alabama", "region": "South"},
        {"state_code": "AK", "state_name": "Alaska", "region": "West"},
        {"state_code": "AZ", "state_name": "Arizona", "region": "West"},
        {"state_code": "AR", "state_name": "Arkansas", "region": "South"},
        {"state_code": "CA", "state_name": "California", "region": "West"},
        {"state_code": "CO", "state_name": "Colorado", "region": "West"},
        {"state_code": "CT", "state_name": "Connecticut", "region": "Northeast"},
        {"state_code": "DE", "state_name": "Delaware", "region": "Northeast"},
        {"state_code": "FL", "state_name": "Florida", "region": "South"},
        {"state_code": "GA", "state_name": "Georgia", "region": "South"},
        {"state_code": "HI", "state_name": "Hawaii", "region": "West"},
        {"state_code": "ID", "state_name": "Idaho", "region": "West"},
        {"state_code": "IL", "state_name": "Illinois", "region": "Midwest"},
        {"state_code": "IN", "state_name": "Indiana", "region": "Midwest"},
        {"state_code": "IA", "state_name": "Iowa", "region": "Midwest"},
        {"state_code": "KS", "state_name": "Kansas", "region": "Midwest"},
        {"state_code": "KY", "state_name": "Kentucky", "region": "South"},
        {"state_code": "LA", "state_name": "Louisiana", "region": "South"},
        {"state_code": "ME", "state_name": "Maine", "region": "Northeast"},
        {"state_code": "MD", "state_name": "Maryland", "region": "Northeast"},
        {"state_code": "MA", "state_name": "Massachusetts", "region": "Northeast"},
        {"state_code": "MI", "state_name": "Michigan", "region": "Midwest"},
        {"state_code": "MN", "state_name": "Minnesota", "region": "Midwest"},
        {"state_code": "MS", "state_name": "Mississippi", "region": "South"},
        {"state_code": "MO", "state_name": "Missouri", "region": "Midwest"},
        {"state_code": "MT", "state_name": "Montana", "region": "West"},
        {"state_code": "NE", "state_name": "Nebraska", "region": "Midwest"},
        {"state_code": "NV", "state_name": "Nevada", "region": "West"},
        {"state_code": "NH", "state_name": "New Hampshire", "region": "Northeast"},
        {"state_code": "NJ", "state_name": "New Jersey", "region": "Northeast"},
        {"state_code": "NM", "state_name": "New Mexico", "region": "West"},
        {"state_code": "NY", "state_name": "New York", "region": "Northeast"},
        {"state_code": "NC", "state_name": "North Carolina", "region": "South"},
        {"state_code": "ND", "state_name": "North Dakota", "region": "Midwest"},
        {"state_code": "OH", "state_name": "Ohio", "region": "Midwest"},
        {"state_code": "OK", "state_name": "Oklahoma", "region": "South"},
        {"state_code": "OR", "state_name": "Oregon", "region": "West"},
        {"state_code": "PA", "state_name": "Pennsylvania", "region": "Northeast"},
        {"state_code": "RI", "state_name": "Rhode Island", "region": "Northeast"},
        {"state_code": "SC", "state_name": "South Carolina", "region": "South"},
        {"state_code": "SD", "state_name": "South Dakota", "region": "Midwest"},
        {"state_code": "TN", "state_name": "Tennessee", "region": "South"},
        {"state_code": "TX", "state_name": "Texas", "region": "South"},
        {"state_code": "UT", "state_name": "Utah", "region": "West"},
        {"state_code": "VT", "state_name": "Vermont", "region": "Northeast"},
        {"state_code": "VA", "state_name": "Virginia", "region": "South"},
        {"state_code": "WA", "state_name": "Washington", "region": "West"},
        {"state_code": "WV", "state_name": "West Virginia", "region": "South"},
        {"state_code": "WI", "state_name": "Wisconsin", "region": "Midwest"},
        {"state_code": "WY", "state_name": "Wyoming", "region": "West"},
        {"state_code": "DC", "state_name": "District of Columbia", "region": "Northeast"},
    ]
    df = pd.DataFrame(us_states)
    df.to_csv(STATIC_DIR / "state_codes.csv", index=False)
    print(f"✅ Generated state_codes.csv with {len(df)} records")
    return df


def generate_zip_codes(num_records=50000):
    """Generate zip codes dimension table"""
    zip_codes = []
    for _ in range(num_records):
        zip_code = fake.zipcode()
        city = fake.city()
        state_code = fake.state_abbr()
        zip_codes.append({
            "zip_code": zip_code,
            "city": city,
            "state_code": state_code,
            "latitude": round(fake.latitude(), 6),
            "longitude": round(fake.longitude(), 6),
            "timezone": fake.timezone(),
        })
    
    df = pd.DataFrame(zip_codes)
    df = df.drop_duplicates(subset=['zip_code'])
    df.to_csv(STATIC_DIR / "zip_codes.csv", index=False)
    print(f"✅ Generated zip_codes.csv with {len(df)} unique records")
    return df


def generate_customers(num_records=1000000, batch_size=None):
    """Generate customer dimension table with 1M records"""
    if batch_size is None:
        batch_size = _default_generate_batch_size()
    print(f"\n📊 Generating {num_records:,} customer records (batch size: {batch_size:,})...")
    
    customers = []
    
    for i in range(0, num_records, batch_size):
        batch_end = min(i + batch_size, num_records)
        batch = []
        
        for j in range(i, batch_end):
            customer_id = f"CUST{str(j+1).zfill(8)}"
            ssn = fake.ssn()
            first_name = fake.first_name()
            last_name = fake.last_name()
            email = f"{first_name.lower()}.{last_name.lower()}@{fake.domain_name()}"
            phone = fake.phone_number()
            date_of_birth = fake.date_of_birth(minimum_age=18, maximum_age=90)
            account_opened_date = fake.date_between(start_date='-10y', end_date='today')
            
            # Address
            street_address = fake.street_address()
            city = fake.city()
            state_code = fake.state_abbr()
            zip_code = fake.zipcode()
            country_code = "US"  # Most customers are US-based
            
            # Customer attributes
            customer_type = np.random.choice(
                ["Individual", "Business", "Premium", "VIP"],
                p=[0.6, 0.25, 0.1, 0.05]
            )
            account_status = np.random.choice(
                ["Active", "Inactive", "Closed", "Suspended"],
                p=[0.85, 0.05, 0.08, 0.02]
            )
            credit_score = np.random.randint(300, 850) if customer_type in ["Individual", "Premium", "VIP"] else None
            
            batch.append({
                "customer_id": customer_id,
                "ssn": ssn,
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "phone": phone,
                "date_of_birth": date_of_birth,
                "account_opened_date": account_opened_date,
                "street_address": street_address,
                "city": city,
                "state_code": state_code,
                "zip_code": zip_code,
                "country_code": country_code,
                "customer_type": customer_type,
                "account_status": account_status,
                "credit_score": credit_score,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            })
        
        customers.extend(batch)
        
        if (i + batch_size) % 100000 == 0 or batch_end == num_records:
            print(f"   Progress: {batch_end:,}/{num_records:,} customers generated")
    
    df = pd.DataFrame(customers)
    df.to_csv(DIMENSIONS_DIR / "customers.csv", index=False)
    print(f"✅ Generated customers.csv with {len(df):,} records")
    return df


def generate_ach_transactions(num_records, customer_ids, batch_size=None):
    """Generate ACH (Automated Clearing House) transactions"""
    if batch_size is None:
        batch_size = _default_generate_batch_size()
    print(f"\n📊 Generating {num_records:,} ACH transaction records (batch size: {batch_size:,})...")
    
    transactions = []
    transaction_types = ["Debit", "Credit"]
    ach_codes = ["PPD", "WEB", "TEL", "CCD", "ARC", "BOC", "POP"]
    
    for i in range(0, num_records, batch_size):
        batch_end = min(i + batch_size, num_records)
        batch = []
        
        for j in range(i, batch_end):
            transaction_id = f"ACH{str(j+1).zfill(10)}"
            customer_id = np.random.choice(customer_ids)
            transaction_datetime = fake.date_time_between(start_date='-2y', end_date='now')
            transaction_date = transaction_datetime.date()
            
            transaction_type = np.random.choice(transaction_types)
            amount = round(np.random.lognormal(mean=5, sigma=1.5), 2)
            if transaction_type == "Debit":
                amount = -abs(amount)
            else:
                amount = abs(amount)
            
            ach_code = np.random.choice(ach_codes)
            # Generate routing number (9 digits, US openInt routing format)
            routing_number = f"{np.random.randint(100000000, 999999999)}"
            account_number = fake.bban()
            
            status = np.random.choice(
                ["Completed", "Pending", "Failed", "Reversed"],
                p=[0.92, 0.05, 0.02, 0.01]
            )
            
            batch.append({
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "transaction_date": transaction_date,
                "transaction_datetime": transaction_datetime,
                "transaction_type": transaction_type,
                "amount": amount,
                "currency": "USD",
                "ach_code": ach_code,
                "routing_number": routing_number,
                "account_number": account_number,
                "description": f"ACH {transaction_type} - {ach_code}",
                "status": status,
                "created_at": datetime.now(),
            })
        
        transactions.extend(batch)
        
        if (i + batch_size) % 100000 == 0 or batch_end == num_records:
            print(f"   Progress: {batch_end:,}/{num_records:,} ACH transactions generated")
    
    df = pd.DataFrame(transactions)
    df.to_csv(FACTS_DIR / "ach_transactions.csv", index=False)
    print(f"✅ Generated ach_transactions.csv with {len(df):,} records")
    return df


def generate_wire_transactions(num_records, customer_ids, batch_size=None):
    """Generate Wire transfer transactions"""
    if batch_size is None:
        batch_size = _default_generate_batch_size()
    print(f"\n📊 Generating {num_records:,} Wire transaction records (batch size: {batch_size:,})...")
    
    transactions = []
    wire_types = ["Domestic", "International"]
    countries = ["US", "CA", "GB", "DE", "FR", "JP", "AU", "CH"]
    
    for i in range(0, num_records, batch_size):
        batch_end = min(i + batch_size, num_records)
        batch = []
        
        for j in range(i, batch_end):
            transaction_id = f"WIRE{str(j+1).zfill(10)}"
            customer_id = np.random.choice(customer_ids)
            transaction_datetime = fake.date_time_between(start_date='-2y', end_date='now')
            transaction_date = transaction_datetime.date()
            
            wire_type = np.random.choice(wire_types, p=[0.7, 0.3])
            amount = round(np.random.lognormal(mean=7, sigma=1.8), 2)  # Wires are typically larger
            
            if wire_type == "International":
                currency = np.random.choice(["USD", "EUR", "GBP", "JPY", "CAD", "AUD"])
                beneficiary_country = np.random.choice(countries)
                beneficiary_openInt_swift = fake.swift()
            else:
                currency = "USD"
                beneficiary_country = "US"
                beneficiary_openInt_swift = None
            
            # Generate routing numbers (9 digits, US openInt routing format)
            sender_routing = f"{np.random.randint(100000000, 999999999)}"
            beneficiary_routing = f"{np.random.randint(100000000, 999999999)}" if wire_type == "Domestic" else None
            beneficiary_account = fake.bban()
            beneficiary_name = fake.name()
            
            status = np.random.choice(
                ["Completed", "Pending", "Failed", "Cancelled"],
                p=[0.88, 0.08, 0.03, 0.01]
            )
            
            fee = round(amount * 0.001, 2) if wire_type == "Domestic" else round(amount * 0.002, 2)
            
            batch.append({
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "transaction_date": transaction_date,
                "transaction_datetime": transaction_datetime,
                "wire_type": wire_type,
                "amount": amount,
                "currency": currency,
                "sender_routing": sender_routing,
                "beneficiary_routing": beneficiary_routing,
                "beneficiary_account": beneficiary_account,
                "beneficiary_name": beneficiary_name,
                "beneficiary_country": beneficiary_country,
                "beneficiary_openInt_swift": beneficiary_openInt_swift,
                "fee": fee,
                "description": f"Wire Transfer - {wire_type}",
                "status": status,
                "created_at": datetime.now(),
            })
        
        transactions.extend(batch)
        
        if (i + batch_size) % 100000 == 0 or batch_end == num_records:
            print(f"   Progress: {batch_end:,}/{num_records:,} Wire transactions generated")
    
    df = pd.DataFrame(transactions)
    df.to_csv(FACTS_DIR / "wire_transactions.csv", index=False)
    print(f"✅ Generated wire_transactions.csv with {len(df):,} records")
    return df


def generate_credit_transactions(num_records, customer_ids, batch_size=None):
    """Generate Credit card transactions"""
    if batch_size is None:
        batch_size = _default_generate_batch_size()
    print(f"\n📊 Generating {num_records:,} Credit card transaction records (batch size: {batch_size:,})...")
    
    transactions = []
    card_types = ["Visa", "Mastercard", "American Express", "Discover"]
    merchant_categories = [
        "Retail", "Restaurant", "Gas Station", "Grocery", "Online Shopping",
        "Travel", "Entertainment", "Healthcare", "Utilities", "Education"
    ]
    
    for i in range(0, num_records, batch_size):
        batch_end = min(i + batch_size, num_records)
        batch = []
        
        for j in range(i, batch_end):
            transaction_id = f"CREDIT{str(j+1).zfill(10)}"
            customer_id = np.random.choice(customer_ids)
            transaction_datetime = fake.date_time_between(start_date='-1y', end_date='now')
            transaction_date = transaction_datetime.date()
            
            card_type = np.random.choice(card_types)
            card_number_last4 = str(np.random.randint(1000, 9999))
            amount = round(np.random.lognormal(mean=4, sigma=1.2), 2)
            
            merchant_name = fake.company()
            merchant_category = np.random.choice(merchant_categories)
            merchant_city = fake.city()
            merchant_state = fake.state_abbr()
            merchant_country = "US"
            
            authorization_code = fake.bothify(text='??????', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            status = np.random.choice(
                ["Approved", "Declined", "Pending", "Refunded"],
                p=[0.94, 0.04, 0.01, 0.01]
            )
            
            batch.append({
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "transaction_date": transaction_date,
                "transaction_datetime": transaction_datetime,
                "card_type": card_type,
                "card_number_last4": card_number_last4,
                "amount": amount,
                "currency": "USD",
                "merchant_name": merchant_name,
                "merchant_category": merchant_category,
                "merchant_city": merchant_city,
                "merchant_state": merchant_state,
                "merchant_country": merchant_country,
                "authorization_code": authorization_code,
                "description": f"{merchant_category} - {merchant_name}",
                "status": status,
                "created_at": datetime.now(),
            })
        
        transactions.extend(batch)
        
        if (i + batch_size) % 100000 == 0 or batch_end == num_records:
            print(f"   Progress: {batch_end:,}/{num_records:,} Credit transactions generated")
    
    df = pd.DataFrame(transactions)
    df.to_csv(FACTS_DIR / "credit_transactions.csv", index=False)
    print(f"✅ Generated credit_transactions.csv with {len(df):,} records")
    return df


def generate_check_transactions(num_records, customer_ids, batch_size=None):
    """Generate Check transactions"""
    if batch_size is None:
        batch_size = _default_generate_batch_size()
    print(f"\n📊 Generating {num_records:,} Check transaction records (batch size: {batch_size:,})...")
    
    transactions = []
    
    for i in range(0, num_records, batch_size):
        batch_end = min(i + batch_size, num_records)
        batch = []
        
        for j in range(i, batch_end):
            transaction_id = f"CHECK{str(j+1).zfill(10)}"
            customer_id = np.random.choice(customer_ids)
            transaction_datetime = fake.date_time_between(start_date='-2y', end_date='now')
            transaction_date = transaction_datetime.date()
            
            check_number = np.random.randint(100, 999999)
            amount = round(np.random.lognormal(mean=5.5, sigma=1.4), 2)
            
            payee_name = fake.name()
            memo = np.random.choice([
                "Payment", "Rent", "Utilities", "Services", "Invoice Payment",
                "Loan Payment", "Insurance", "Tax Payment"
            ], p=[0.2, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05])
            
            status = np.random.choice(
                ["Cleared", "Pending", "Bounced", "Cancelled"],
                p=[0.85, 0.1, 0.04, 0.01]
            )
            
            batch.append({
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "transaction_date": transaction_date,
                "transaction_datetime": transaction_datetime,
                "check_number": check_number,
                "amount": amount,
                "currency": "USD",
                "payee_name": payee_name,
                "memo": memo,
                "description": f"Check #{check_number} - {payee_name}",
                "status": status,
                "created_at": datetime.now(),
            })
        
        transactions.extend(batch)
        
        if (i + batch_size) % 100000 == 0 or batch_end == num_records:
            print(f"   Progress: {batch_end:,}/{num_records:,} Check transactions generated")
    
    df = pd.DataFrame(transactions)
    df.to_csv(FACTS_DIR / "check_transactions.csv", index=False)
    print(f"✅ Generated check_transactions.csv with {len(df):,} records")
    return df


def generate_debit_transactions(num_records, customer_ids, batch_size=None):
    """Generate Debit card transactions"""
    if batch_size is None:
        batch_size = _default_generate_batch_size()
    print(f"\n📊 Generating {num_records:,} Debit card transaction records (batch size: {batch_size:,})...")
    
    transactions = []
    transaction_types = ["Purchase", "ATM Withdrawal", "Online Payment", "Recurring Payment"]
    merchant_categories = [
        "Retail", "Restaurant", "Gas Station", "Grocery", "ATM",
        "Online Shopping", "Utilities", "Subscription"
    ]
    
    for i in range(0, num_records, batch_size):
        batch_end = min(i + batch_size, num_records)
        batch = []
        
        for j in range(i, batch_end):
            transaction_id = f"DEBIT{str(j+1).zfill(10)}"
            customer_id = np.random.choice(customer_ids)
            transaction_datetime = fake.date_time_between(start_date='-1y', end_date='now')
            transaction_date = transaction_datetime.date()
            
            transaction_type = np.random.choice(transaction_types)
            amount = round(np.random.lognormal(mean=3.8, sigma=1.1), 2)
            amount = -abs(amount)  # Debits are negative
            
            if transaction_type == "ATM Withdrawal":
                merchant_name = "ATM"
                merchant_category = "ATM"
                merchant_location = fake.address()
            else:
                merchant_name = fake.company()
                merchant_category = np.random.choice(merchant_categories)
                merchant_location = None
            
            status = np.random.choice(
                ["Completed", "Pending", "Declined", "Reversed"],
                p=[0.93, 0.04, 0.02, 0.01]
            )
            
            batch.append({
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "transaction_date": transaction_date,
                "transaction_datetime": transaction_datetime,
                "transaction_type": transaction_type,
                "amount": amount,
                "currency": "USD",
                "merchant_name": merchant_name,
                "merchant_category": merchant_category,
                "merchant_location": merchant_location,
                "description": f"{transaction_type} - {merchant_name}",
                "status": status,
                "created_at": datetime.now(),
            })
        
        transactions.extend(batch)
        
        if (i + batch_size) % 100000 == 0 or batch_end == num_records:
            print(f"   Progress: {batch_end:,}/{num_records:,} Debit transactions generated")
    
    df = pd.DataFrame(transactions)
    df.to_csv(FACTS_DIR / "debit_transactions.csv", index=False)
    print(f"✅ Generated debit_transactions.csv with {len(df):,} records")
    return df


def main():
    """Main function to generate all test data"""
    parser = argparse.ArgumentParser(
        description="Generate openInt test data (customers, transactions, static dimension tables)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all data (default)
  python generate_openInt_test_data.py

  # Generate only one category
  python generate_openInt_test_data.py --only-customers
  python generate_openInt_test_data.py --only-transactions
  python generate_openInt_test_data.py --only-static

  # With custom counts
  python generate_openInt_test_data.py --only-customers --num-customers 50000
  python generate_openInt_test_data.py --only-transactions --num-transactions 100000
        """
    )
    only_group = parser.add_mutually_exclusive_group()
    only_group.add_argument(
        "--only-customers",
        action="store_true",
        help="Generate only customers (dimensions/customers.csv); skip transactions and static"
    )
    only_group.add_argument(
        "--only-transactions",
        action="store_true",
        help="Generate only transaction fact tables; requires existing customers.csv"
    )
    only_group.add_argument(
        "--only-static",
        action="store_true",
        help="Generate only static tables (country_codes, state_codes, zip_codes); skip customers and transactions"
    )
    parser.add_argument(
        "--num-customers",
        type=int,
        default=10_000,
        help="Number of customer records to generate (default: 10,000)"
    )
    parser.add_argument(
        "--num-transactions",
        type=int,
        default=100_000,
        help="Total number of transactions to generate across all types, drawn from the customer set (default: 100,000)"
    )
    default_batch = _default_generate_batch_size()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_batch,
        help=f"In-memory batch size for generation loops (default: {default_batch:,}, from system; 10K–50K; set GENERATE_BATCH_SIZE to override)"
    )
    args = parser.parse_args()

    load_static = args.only_static or (not args.only_customers and not args.only_transactions)
    load_customers = args.only_customers or (not args.only_static and not args.only_transactions)
    load_transactions = args.only_transactions or (not args.only_static and not args.only_customers)

    print("=" * 80)
    print("🏦 openInt Test Data Generator")
    print("=" * 80)
    if args.only_customers:
        print("   📌 Generating only: customers")
    elif args.only_transactions:
        print("   📌 Generating only: transactions")
    elif args.only_static:
        print("   📌 Generating only: static (country, state, zip)")
    print(f"   📦 Batch size: {args.batch_size:,} (10K–50K based on system)")
    print()

    start_time = datetime.now()

    # Generate static dimension tables
    if load_static:
        print("\n📋 Generating static dimension tables...")
        generate_country_codes()
        generate_state_codes()
        generate_zip_codes()

    # Generate customers and/or get customer_ids for transactions
    customer_ids = None
    if load_customers:
        customers_df = generate_customers(num_records=args.num_customers, batch_size=args.batch_size)
        customer_ids = customers_df['customer_id'].tolist()
    elif load_transactions:
        # Transactions need customer_ids from existing customers.csv
        customers_path = DIMENSIONS_DIR / "customers.csv"
        if not customers_path.exists():
            print(f"\n❌ Cannot generate only transactions: {customers_path} not found.")
            print("   Generate customers first: python generate_openInt_test_data.py --only-customers")
            sys.exit(1)
        customers_df = pd.read_csv(customers_path)
        customer_ids = customers_df['customer_id'].tolist()
        print(f"\n📂 Loaded {len(customer_ids):,} customer IDs from {customers_path}")

    # Generate transaction fact tables
    transaction_distribution = None
    if load_transactions and customer_ids is not None:
        print("\n💳 Generating transaction fact tables...")
        total_transactions = args.num_transactions
        transaction_distribution = {
            "credit": int(total_transactions * 0.35),
            "debit": int(total_transactions * 0.30),
            "ach": int(total_transactions * 0.20),
            "wire": int(total_transactions * 0.10),
            "check": int(total_transactions * 0.05),
        }
        generate_credit_transactions(transaction_distribution["credit"], customer_ids, args.batch_size)
        generate_debit_transactions(transaction_distribution["debit"], customer_ids, args.batch_size)
        generate_ach_transactions(transaction_distribution["ach"], customer_ids, args.batch_size)
        generate_wire_transactions(transaction_distribution["wire"], customer_ids, args.batch_size)
        generate_check_transactions(transaction_distribution["check"], customer_ids, args.batch_size)

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("✅ Test Data Generation Complete!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    if load_static:
        print(f"   • Static: country_codes, state_codes, zip_codes")
    if load_customers:
        print(f"   • Customer records: {args.num_customers:,}")
    if load_transactions and transaction_distribution:
        print(f"   • Credit transactions: {transaction_distribution['credit']:,}")
        print(f"   • Debit transactions: {transaction_distribution['debit']:,}")
        print(f"   • ACH transactions: {transaction_distribution['ach']:,}")
        print(f"   • Wire transactions: {transaction_distribution['wire']:,}")
        print(f"   • Check transactions: {transaction_distribution['check']:,}")
        print(f"   • Total transactions: {args.num_transactions:,}")
    print(f"\n📁 Data saved to:")
    print(f"   • Dimensions: {DIMENSIONS_DIR}")
    print(f"   • Facts: {FACTS_DIR}")
    print(f"   • Static: {STATIC_DIR}")
    print(f"\n⏱️  Generation time: {duration}")
    print("=" * 80)


if __name__ == "__main__":
    main()
