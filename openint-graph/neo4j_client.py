"""
Neo4j client for OpenInt agent system.
Connects via Bolt, runs parameterized Cypher, returns list of dicts or path DTOs.
"""

import os
from typing import List, Dict, Any, Optional

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None


class Neo4jClient:
    """
    Neo4j client for relationship and path queries.
    Defaults: bolt://localhost:7687, neo4j/datahub, database graph.db (Neo4j with DataHub on Docker).
    Override via NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """
        Initialize Neo4j client.

        Args:
            uri: Bolt URI (default NEO4J_URI or bolt://localhost:7687)
            user: Username (default NEO4J_USER or neo4j)
            password: Password (default NEO4J_PASSWORD or datahub)
            database: Database name (default NEO4J_DATABASE or neo4j; use graph.db for DataHub compose)
        """
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j driver not available. Install with: pip install neo4j")
        self._uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "datahub")
        self._database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self._driver = None

    def connect(self) -> None:
        """Create driver (lazy)."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )

    def close(self) -> None:
        """Close driver."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jClient":
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def run(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run a Cypher query and return list of record dicts.

        Args:
            cypher: Cypher query (parameterized with $param)
            parameters: Query parameters

        Returns:
            List of dicts, one per record (keys = result columns)
        """
        self.connect()
        parameters = parameters or {}
        eager = self._driver.execute_query(
            cypher,
            parameters_=parameters,
            database_=self._database,
        )
        # EagerResult has .records (list of Record); Record.data() -> dict
        return [rec.data() for rec in eager.records]

    def verify_connectivity(self) -> bool:
        """Verify connection to Neo4j. Returns True if ok."""
        try:
            self.connect()
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    def delete_all(self) -> None:
        """Delete all nodes and relationships in the database. Use with caution."""
        self.run("MATCH (n) DETACH DELETE n")


def get_neo4j_client(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> Optional[Neo4jClient]:
    """
    Get a configured Neo4j client, or None if driver unavailable.

    Args:
        uri: Optional override for NEO4J_URI
        user: Optional override for NEO4J_USER
        password: Optional override for NEO4J_PASSWORD

    Returns:
        Neo4jClient instance or None
    """
    if not NEO4J_AVAILABLE:
        return None
    return Neo4jClient(uri=uri, user=user, password=password)
