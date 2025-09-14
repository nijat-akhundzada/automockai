from faker import Faker
from faker.providers import BaseProvider
import random
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CustomProvider(BaseProvider):
    """Custom Faker provider for specialized data types."""
    
    def business_email(self):
        """Generate business email addresses."""
        domains = ['company.com', 'business.org', 'enterprise.net', 'corp.com']
        return f"{self.generator.user_name()}@{random.choice(domains)}"
    
    def product_name(self):
        """Generate product names."""
        adjectives = ['Premium', 'Professional', 'Advanced', 'Ultimate', 'Standard', 'Basic']
        nouns = ['Solution', 'System', 'Platform', 'Tool', 'Service', 'Product']
        return f"{random.choice(adjectives)} {random.choice(nouns)}"
    
    def currency_amount(self, min_value=1.0, max_value=10000.0):
        """Generate currency amounts with proper decimal places."""
        return round(random.uniform(min_value, max_value), 2)
    
    def database_id(self):
        """Generate database-style IDs."""
        return random.randint(1, 999999)


class FallbackGenerator:
    """
    Fallback Agent - Guarantees type correctness when AI fails.
    Uses Faker with custom providers and supports deterministic results with seed.
    """
    
    def __init__(self, seed: Optional[int] = None, locale: str = 'en_US'):
        self.seed = seed
        self.fake = Faker(locale)
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)
        
        # Add custom provider
        self.fake.add_provider(CustomProvider)
        
        # Type mapping for SQL types to Faker methods
        self.type_mapping = {
            # String types
            'varchar': self._generate_varchar,
            'char': self._generate_char,
            'text': self._generate_text,
            'string': self._generate_text,
            
            # Numeric types
            'integer': self._generate_integer,
            'int': self._generate_integer,
            'bigint': self._generate_bigint,
            'smallint': self._generate_smallint,
            'decimal': self._generate_decimal,
            'numeric': self._generate_decimal,
            'float': self._generate_float,
            'real': self._generate_float,
            'double': self._generate_float,
            
            # Date/Time types
            'date': self._generate_date,
            'time': self._generate_time,
            'timestamp': self._generate_timestamp,
            'datetime': self._generate_datetime,
            
            # Boolean
            'boolean': self._generate_boolean,
            'bool': self._generate_boolean,
            
            # JSON/JSONB
            'json': self._generate_json,
            'jsonb': self._generate_json,
            
            # UUID
            'uuid': self._generate_uuid,
        }
        
        # Semantic type mapping
        self.semantic_mapping = {
            'email': self._generate_email,
            'phone': self._generate_phone,
            'name': self._generate_name,
            'address': self._generate_address,
            'money': self._generate_money,
            'url': self._generate_url,
            'datetime': self._generate_datetime,
            'text': self._generate_description,
        }
    
    def generate_column_value(self, column_info: Dict[str, Any]) -> Any:
        """Generate a value for a column based on its type and constraints."""
        col_type = column_info["type"].lower()
        col_name = column_info["name"].lower()
        
        # Handle nullable columns
        if column_info.get("nullable", True) and random.random() < 0.1:  # 10% chance of NULL
            return None
        
        # Check for semantic meaning in column name first
        semantic_type = self._detect_semantic_type(col_name, col_type)
        if semantic_type and semantic_type in self.semantic_mapping:
            return self.semantic_mapping[semantic_type](column_info)
        
        # Fall back to type-based generation
        for type_key, generator in self.type_mapping.items():
            if type_key in col_type:
                return generator(column_info)
        
        # Default fallback
        return self._generate_text(column_info)
    
    def generate_semantic_value(self, column_info: Dict[str, Any], semantic_type: str, 
                              constraints: Dict[str, Any] = None) -> Any:
        """Generate a value for a specific semantic type."""
        if semantic_type in self.semantic_mapping:
            return self.semantic_mapping[semantic_type](column_info, constraints)
        return self._generate_text(column_info)
    
    def _detect_semantic_type(self, col_name: str, col_type: str) -> Optional[str]:
        """Detect semantic type from column name and SQL type."""
        col_name = col_name.lower()
        
        if 'email' in col_name:
            return 'email'
        elif any(name_part in col_name for name_part in ['name', 'title', 'label']):
            return 'name'
        elif 'phone' in col_name or 'mobile' in col_name:
            return 'phone'
        elif any(addr_part in col_name for addr_part in ['address', 'street', 'city', 'country']):
            return 'address'
        elif any(money_part in col_name for money_part in ['price', 'cost', 'amount', 'salary', 'wage']):
            return 'money'
        elif 'url' in col_name or 'link' in col_name:
            return 'url'
        elif any(text_part in col_name for text_part in ['description', 'comment', 'note', 'bio']):
            return 'text'
        elif any(date_part in col_type for date_part in ['date', 'time', 'timestamp']):
            return 'datetime'
        
        return None
    
    # Type-specific generators
    def _generate_varchar(self, column_info: Dict[str, Any]) -> str:
        """Generate VARCHAR values respecting length constraints."""
        max_length = column_info.get("length", 255)
        text = self.fake.text(max_nb_chars=min(max_length, 100))
        return text[:max_length] if len(text) > max_length else text
    
    def _generate_char(self, column_info: Dict[str, Any]) -> str:
        """Generate CHAR values with exact length."""
        length = column_info.get("length", 10)
        text = self.fake.lexify('?' * length)
        return text[:length].ljust(length)
    
    def _generate_text(self, column_info: Dict[str, Any]) -> str:
        """Generate TEXT values."""
        return self.fake.text(max_nb_chars=500)
    
    def _generate_integer(self, column_info: Dict[str, Any]) -> int:
        """Generate INTEGER values."""
        return self.fake.random_int(min=-2147483648, max=2147483647)
    
    def _generate_bigint(self, column_info: Dict[str, Any]) -> int:
        """Generate BIGINT values."""
        return self.fake.random_int(min=-9223372036854775808, max=9223372036854775807)
    
    def _generate_smallint(self, column_info: Dict[str, Any]) -> int:
        """Generate SMALLINT values."""
        return self.fake.random_int(min=-32768, max=32767)
    
    def _generate_decimal(self, column_info: Dict[str, Any]) -> float:
        """Generate DECIMAL/NUMERIC values."""
        precision = column_info.get("precision", 10)
        scale = column_info.get("scale", 2)
        
        # Generate a number with appropriate precision and scale
        max_digits = precision - scale
        integer_part = random.randint(0, 10**max_digits - 1) if max_digits > 0 else 0
        decimal_part = random.randint(0, 10**scale - 1) if scale > 0 else 0
        
        return float(f"{integer_part}.{decimal_part:0{scale}d}")
    
    def _generate_float(self, column_info: Dict[str, Any]) -> float:
        """Generate FLOAT/REAL/DOUBLE values."""
        return self.fake.pyfloat(left_digits=5, right_digits=2, positive=True)
    
    def _generate_date(self, column_info: Dict[str, Any]) -> str:
        """Generate DATE values."""
        return self.fake.date_between(start_date='-5y', end_date='today').strftime('%Y-%m-%d')
    
    def _generate_time(self, column_info: Dict[str, Any]) -> str:
        """Generate TIME values."""
        return self.fake.time()
    
    def _generate_timestamp(self, column_info: Dict[str, Any]) -> str:
        """Generate TIMESTAMP values."""
        return self.fake.date_time_between(start_date='-2y', end_date='now').strftime('%Y-%m-%d %H:%M:%S')
    
    def _generate_datetime(self, column_info: Dict[str, Any], constraints: Dict[str, Any] = None) -> str:
        """Generate DATETIME values."""
        return self.fake.date_time_between(start_date='-1y', end_date='now').strftime('%Y-%m-%d %H:%M:%S')
    
    def _generate_boolean(self, column_info: Dict[str, Any]) -> bool:
        """Generate BOOLEAN values."""
        return self.fake.boolean()
    
    def _generate_json(self, column_info: Dict[str, Any]) -> str:
        """Generate JSON values."""
        data = {
            "id": self.fake.random_int(1, 1000),
            "name": self.fake.name(),
            "active": self.fake.boolean(),
            "created_at": self.fake.iso8601()
        }
        return str(data).replace("'", '"')
    
    def _generate_uuid(self, column_info: Dict[str, Any]) -> str:
        """Generate UUID values."""
        return str(self.fake.uuid4())
    
    # Semantic generators
    def _generate_email(self, column_info: Dict[str, Any], constraints: Dict[str, Any] = None) -> str:
        """Generate email addresses."""
        if 'business' in column_info["name"].lower() or 'company' in column_info["name"].lower():
            return self.fake.business_email()
        return self.fake.email()
    
    def _generate_phone(self, column_info: Dict[str, Any], constraints: Dict[str, Any] = None) -> str:
        """Generate phone numbers."""
        return self.fake.phone_number()
    
    def _generate_name(self, column_info: Dict[str, Any], constraints: Dict[str, Any] = None) -> str:
        """Generate names based on context."""
        col_name = column_info["name"].lower()
        
        if 'first' in col_name:
            return self.fake.first_name()
        elif 'last' in col_name:
            return self.fake.last_name()
        elif 'company' in col_name or 'business' in col_name:
            return self.fake.company()
        elif 'product' in col_name:
            return self.fake.product_name()
        else:
            return self.fake.name()
    
    def _generate_address(self, column_info: Dict[str, Any], constraints: Dict[str, Any] = None) -> str:
        """Generate address components."""
        col_name = column_info["name"].lower()
        
        if 'street' in col_name:
            return self.fake.street_address()
        elif 'city' in col_name:
            return self.fake.city()
        elif 'state' in col_name:
            return self.fake.state()
        elif 'country' in col_name:
            return self.fake.country()
        elif 'zip' in col_name or 'postal' in col_name:
            return self.fake.zipcode()
        else:
            return self.fake.address()
    
    def _generate_money(self, column_info: Dict[str, Any], constraints: Dict[str, Any] = None) -> float:
        """Generate monetary values."""
        col_name = column_info["name"].lower()
        
        if 'salary' in col_name or 'wage' in col_name:
            return self.fake.currency_amount(30000, 200000)
        elif 'price' in col_name:
            return self.fake.currency_amount(1, 1000)
        else:
            return self.fake.currency_amount(1, 10000)
    
    def _generate_url(self, column_info: Dict[str, Any], constraints: Dict[str, Any] = None) -> str:
        """Generate URLs."""
        return self.fake.url()
    
    def _generate_description(self, column_info: Dict[str, Any], constraints: Dict[str, Any] = None) -> str:
        """Generate descriptive text."""
        max_length = column_info.get("length", 500)
        text = self.fake.text(max_nb_chars=min(max_length, 500))
        return text[:max_length] if len(text) > max_length else text
    
    def generate_fake_rows(self, table_name: str, table_info: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
        """Generate multiple rows of fake data for a table."""
        rows = []
        columns = table_info["columns"]
        pk_columns = set(table_info.get("primary_key", {}).get("constrained_columns", []))
        
        for i in range(count):
            row = {}
            for column in columns:
                col_name = column["name"]
                
                # Skip auto-increment primary keys
                if (col_name in pk_columns and 
                    column.get("autoincrement", False) and 
                    any(kw in column["type"].lower() for kw in ['serial', 'identity', 'autoincrement'])):
                    continue
                
                row[col_name] = self.generate_column_value(column)
            
            rows.append(row)
        
        return rows


# Legacy function for backward compatibility
def generate_fake_rows(table_name: str, table_info: Dict[str, Any], count: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Legacy function for backward compatibility."""
    generator = FallbackGenerator(seed=seed)
    return generator.generate_fake_rows(table_name, table_info, count)