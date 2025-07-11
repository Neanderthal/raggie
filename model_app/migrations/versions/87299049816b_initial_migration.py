"""Initial Migration

Revision ID: 87299049816b
Revises: 
Create Date: 2025-06-17 23:55:37.524018

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel
from sqlalchemy.dialects import postgresql
import pgvector.sqlalchemy

# revision identifiers, used by Alembic.
revision = '87299049816b'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    # Drop dependent table first
    op.drop_index(op.f('ix_cmetadata_gin'), table_name='langchain_pg_embedding', postgresql_using='gin')
    op.drop_index(op.f('ix_langchain_pg_embedding_collection_id'), table_name='langchain_pg_embedding')
    op.drop_table('langchain_pg_embedding')
    # Now drop the parent table
    op.drop_table('langchain_pg_collection')
    op.alter_column('documents', 'content',
               existing_type=sa.TEXT(),
               type_=sa.String(),
               existing_nullable=False)
    op.alter_column('documents', 'embedding',
               existing_type=pgvector.sqlalchemy.Vector(dim=768),
               nullable=True)
    op.alter_column('documents', 'user_id',
               existing_type=sa.INTEGER(),
               nullable=False)
    op.alter_column('documents', 'scope_id',
               existing_type=sa.INTEGER(),
               nullable=False)
    op.drop_index(op.f('idx_documents_scope_id'), table_name='documents')
    op.drop_index(op.f('idx_documents_user_id'), table_name='documents')
    op.drop_column('documents', 'created_at')
    op.alter_column('scopes', 'description',
               existing_type=sa.TEXT(),
               type_=sa.String(),
               existing_nullable=True)
    op.drop_constraint(op.f('scopes_name_key'), 'scopes', type_='unique')
    op.create_index(op.f('ix_scopes_name'), 'scopes', ['name'], unique=True)
    op.drop_column('scopes', 'created_at')
    op.drop_constraint(op.f('users_email_key'), 'users', type_='unique')
    op.drop_constraint(op.f('users_username_key'), 'users', type_='unique')
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    op.drop_column('users', 'created_at')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('users', sa.Column('created_at', postgresql.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=True))
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.create_unique_constraint(op.f('users_username_key'), 'users', ['username'], postgresql_nulls_not_distinct=False)
    op.create_unique_constraint(op.f('users_email_key'), 'users', ['email'], postgresql_nulls_not_distinct=False)
    op.add_column('scopes', sa.Column('created_at', postgresql.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=True))
    op.drop_index(op.f('ix_scopes_name'), table_name='scopes')
    op.create_unique_constraint(op.f('scopes_name_key'), 'scopes', ['name'], postgresql_nulls_not_distinct=False)
    op.alter_column('scopes', 'description',
               existing_type=sa.String(),
               type_=sa.TEXT(),
               existing_nullable=True)
    op.add_column('documents', sa.Column('created_at', postgresql.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=True))
    op.create_index(op.f('idx_documents_user_id'), 'documents', ['user_id'], unique=False)
    op.create_index(op.f('idx_documents_scope_id'), 'documents', ['scope_id'], unique=False)
    op.alter_column('documents', 'scope_id',
               existing_type=sa.INTEGER(),
               nullable=True)
    op.alter_column('documents', 'user_id',
               existing_type=sa.INTEGER(),
               nullable=True)
    op.alter_column('documents', 'embedding',
               existing_type=pgvector.sqlalchemy.Vector(dim=768),
               nullable=False)
    op.alter_column('documents', 'content',
               existing_type=sa.String(),
               type_=sa.TEXT(),
               existing_nullable=False)
    # Create parent table first
    op.create_table('langchain_pg_collection',
    sa.Column('uuid', sa.UUID(), autoincrement=False, nullable=False),
    sa.Column('name', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('cmetadata', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True),
    sa.PrimaryKeyConstraint('uuid', name=op.f('langchain_pg_collection_pkey')),
    sa.UniqueConstraint('name', name=op.f('langchain_pg_collection_name_key'), postgresql_include=[], postgresql_nulls_not_distinct=False)
    )
    # Then create dependent table
    op.create_table('langchain_pg_embedding',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('collection_id', sa.UUID(), autoincrement=False, nullable=True),
    sa.Column('embedding', pgvector.sqlalchemy.Vector(), autoincrement=False, nullable=True),
    sa.Column('document', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('cmetadata', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['collection_id'], ['langchain_pg_collection.uuid'], name=op.f('langchain_pg_embedding_collection_id_fkey'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('langchain_pg_embedding_pkey'))
    )
    op.create_index(op.f('ix_langchain_pg_embedding_collection_id'), 'langchain_pg_embedding', ['collection_id'], unique=False)
    op.create_index(op.f('ix_cmetadata_gin'), 'langchain_pg_embedding', ['cmetadata'], unique=False, postgresql_using='gin')
    # ### end Alembic commands ###
