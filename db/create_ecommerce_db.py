
import sqlite3

def create_ecommerce_db():
    conn = sqlite3.connect('/Users/nijatakhundzada/Desktop/automockai/db/ecommerce.db')
    c = conn.cursor()

    # Create users table
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE
    )
    """)

    # Create products table
    c.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        price REAL NOT NULL
    )
    """)

    # Create orders table
    c.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (product_id) REFERENCES products (id)
    )
    """)

    # Insert sample data
    # users = [
    #     (1, 'Alice', 'alice@example.com'),
    #     (2, 'Bob', 'bob@example.com')
    # ]
    # c.executemany("INSERT OR IGNORE INTO users VALUES (?,?,?)", users)

    # products = [
    #     (1, 'Laptop', 1200.50),
    #     (2, 'Mouse', 25.00),
    #     (3, 'Keyboard', 75.75)
    # ]
    # c.executemany("INSERT OR IGNORE INTO products VALUES (?,?,?)", products)

    # orders = [
    #     (1, 1, 1, 1),
    #     (2, 1, 3, 2),
    #     (3, 2, 2, 1)
    # ]
    # c.executemany("INSERT OR IGNORE INTO orders VALUES (?,?,?,?)", orders)

    

    conn.commit()
    conn.close()

if __name__ == '__main__':
    create_ecommerce_db()
