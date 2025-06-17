# Database Migrations with Alembic and SQLModel

This document explains how to use Alembic migrations with SQLModel in this project.

## Overview

This project uses Alembic for database schema migrations with SQLModel. The migration system allows you to:
- Track database schema changes over time
- Apply incremental updates to the database
- Rollback changes if needed
- Maintain consistency across different environments

## Setup

The migration system is already configured with the following structure:

```
model_app/
├── alembic.ini              # Alembic configuration
├── migrations/
│   ├── env.py              # Migration environment setup
│   ├── script.py.mako      # Migration template
│   └── versions/           # Generated migration files
└── db/
    └── db.py              # SQLModel definitions
```

## Configuration

### Environment Variables

Make sure you have the following environment variables set in `model_app/.env`:

```bash
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name
```

### Database URL

The migration system automatically constructs the database URL from environment variables:
```
postgresql://user:password@host:port/database
```

## Common Migration Commands

All commands should be run from the `model_app` directory:

```bash
cd model_app
```

### 1. Generate a New Migration

When you modify SQLModel classes in `db/db.py`, generate a new migration:

```bash
python -m alembic revision --autogenerate -m "Description of changes"
```

Example:
```bash
python -m alembic revision --autogenerate -m "Add user profile fields"
```

### 2. Apply Migrations

Apply all pending migrations to the database:

```bash
python -m alembic upgrade head
```

### 3. Check Migration Status

See current migration status:

```bash
python -m alembic current
```

### 4. View Migration History

See all migrations:

```bash
python -m alembic history
```

### 5. Rollback Migrations

Rollback to a specific migration:

```bash
python -m alembic downgrade <revision_id>
```

Rollback one migration:

```bash
python -m alembic downgrade -1
```

## Workflow for Schema Changes

### 1. Modify SQLModel Classes

Edit your models in `model_app/db/db.py`. For example:

```python
class User(SQLModel, table=True):
    __tablename__ = "users"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    email: Optional[str] = None
    # Add new field
    full_name: Optional[str] = None  # New field
```

### 2. Generate Migration

```bash
cd model_app
python -m alembic revision --autogenerate -m "Add full_name to User model"
```

### 3. Review Generated Migration

Check the generated file in `migrations/versions/` to ensure it looks correct.

### 4. Apply Migration

```bash
python -m alembic upgrade head
```

## Important Notes

### Adding New Models

When you add new SQLModel classes, make sure to:

1. Import them in `migrations/env.py`:
```python
from db.db import User, Scope, Document, NewModel  # Add NewModel
```

2. Generate and apply the migration as usual.

### Field Constraints

When adding new fields to existing tables:
- Use `Optional` types or provide default values
- This ensures existing data remains valid

```python
# Good - Optional field
new_field: Optional[str] = None

# Good - Field with default
status: str = Field(default="active")

# Avoid - Required field without default (will fail on existing data)
# required_field: str  # This will cause migration issues
```

### Vector Fields

For pgvector fields, use the Column approach:

```python
from sqlalchemy import Column
from pgvector.sqlalchemy import Vector

class Document(SQLModel, table=True):
    embedding: List[float] = Field(sa_column=Column(Vector(768)))
```

### Foreign Keys

Define foreign keys properly:

```python
class Document(SQLModel, table=True):
    user_id: int = Field(foreign_key="users.id")
    scope_id: int = Field(foreign_key="scopes.id")
```

## Troubleshooting

### Migration Conflicts

If you get conflicts when multiple developers create migrations:

1. Pull latest changes
2. Delete your migration file
3. Regenerate the migration
4. Apply it

### Database Connection Issues

If migrations fail to connect:

1. Check your `.env` file has correct database credentials
2. Ensure the database exists
3. Verify the database server is running

### Rollback Issues

If a rollback fails:

1. Check the downgrade function in the migration file
2. You may need to manually fix the downgrade logic
3. Consider creating a new migration to fix issues instead

### Manual Migration Editing

Sometimes you need to edit generated migrations:

1. Review the generated migration file
2. Edit the `upgrade()` and `downgrade()` functions if needed
3. Test the migration on a copy of your data first

## Best Practices

1. **Always review generated migrations** before applying them
2. **Test migrations on a copy of production data** before applying to production
3. **Use descriptive migration messages** that explain what changed
4. **Keep migrations small and focused** - one logical change per migration
5. **Backup your database** before applying migrations in production
6. **Don't edit applied migrations** - create new ones instead

## Example Migration Workflow

```bash
# 1. Make changes to models in db/db.py
# 2. Generate migration
cd model_app
python -m alembic revision --autogenerate -m "Add user preferences table"

# 3. Review the generated file in migrations/versions/
# 4. Apply migration
python -m alembic upgrade head

# 5. Verify changes
python -m alembic current
```

## Production Deployment

For production deployments:

1. **Backup the database** before applying migrations
2. **Test migrations** on a staging environment first
3. **Apply migrations** during maintenance windows if they involve large changes
4. **Monitor** the application after migration deployment

```bash
# Production migration command
cd model_app
python -m alembic upgrade head
```

## Getting Help

- Check Alembic documentation: https://alembic.sqlalchemy.org/
- SQLModel documentation: https://sqlmodel.tiangolo.com/
- For project-specific issues, check the migration files in `migrations/versions/`
