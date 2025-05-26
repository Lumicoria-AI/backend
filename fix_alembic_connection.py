import os

# Read the alembic.ini file
with open('alembic.ini', 'r') as file:
    content = file.read()

# Get database credentials from .env
postgres_user = os.environ.get('POSTGRES_USER', 'postgres')
postgres_password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
postgres_server = os.environ.get('POSTGRES_SERVER', 'localhost')
postgres_port = os.environ.get('POSTGRES_PORT', '5432')
postgres_db = os.environ.get('POSTGRES_DB', 'lumicoria')

# Create the database URL
db_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_server}:{postgres_port}/{postgres_db}"

# Replace the placeholder URL in alembic.ini
updated_content = content.replace("sqlalchemy.url = driver://user:pass@localhost/dbname", f"sqlalchemy.url = {db_url}")

# Write the updated content back to the file
with open('alembic.ini', 'w') as file:
    file.write(updated_content)

print(f"Updated alembic.ini with database URL: {db_url}")