#!/usr/bin/env python
"""
Embedding Engine Benchmark
===========================

Tests embedding generation, semantic search, and hybrid retrieval
on a synthetic 100-file mock repository.

Reports:
- Recall accuracy (synthetic queries)
- Latency (embedding, search)
- Memory usage

Run: python -m saga.benchmarks.embedding_benchmark
"""

import sys
import time
from pathlib import Path

import numpy as np

# Ensure saga is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from saga.core.memory import (
    EdgeType,
    GraphEdge,
    GraphEmbeddingManager,
    GraphNode,
    NodeType,
    RepoGraph,
    SemanticSearchEngine,
)


def generate_mock_repo(num_files: int = 100) -> RepoGraph:
    """
    Generate a synthetic repository with realistic code patterns.

    Structure:
    - num_files files
    - ~3 classes per file
    - ~5 functions per class
    - Import/call relationships
    """
    print(f"Generating mock repository with {num_files} files...")

    graph = RepoGraph()

    # Code pattern templates for semantic diversity
    patterns = [
        ("async_handler", "Async HTTP request handler with error handling", "async def handle(request): try: response = await process(request) except Exception: log_error()"),
        ("data_processor", "Process and validate incoming data", "def process_data(items): for item in items: validate(item); transform(item)"),
        ("cache_manager", "LRU cache with TTL expiration", "class CacheManager: def get(key): if key in cache and not expired: return cache[key]"),
        ("database_query", "Execute SQL query with connection pooling", "async def query(sql): async with pool.acquire() as conn: return await conn.fetch(sql)"),
        ("auth_middleware", "JWT authentication middleware", "def authenticate(token): payload = jwt.decode(token); return User.from_payload(payload)"),
        ("file_uploader", "Chunked file upload with progress", "async def upload(file, chunk_size): for chunk in iter_chunks(file): await send(chunk)"),
        ("websocket_handler", "WebSocket connection handler", "async def ws_handler(ws): async for msg in ws: await process_message(msg)"),
        ("rate_limiter", "Token bucket rate limiting", "def check_rate(key): tokens = bucket.get(key); if tokens > 0: bucket.consume(key)"),
        ("logger_config", "Structured logging configuration", "def setup_logging(): handler = logging.StreamHandler(); formatter = JsonFormatter()"),
        ("config_loader", "YAML configuration with env overrides", "def load_config(path): config = yaml.load(path); apply_env_overrides(config)"),
    ]

    node_count = 0
    function_nodes = []

    for file_idx in range(num_files):
        file_path = f"src/module_{file_idx // 10}/file_{file_idx}.py"
        file_id = f"file:{file_path}"

        # File node
        graph.add_node(GraphNode(
            node_id=file_id,
            node_type=NodeType.FILE,
            name=f"file_{file_idx}.py",
            file_path=file_path,
            metadata={
                "imports": ["asyncio", "logging", "typing"],
                "docstring": f"Module {file_idx} implementing core functionality"
            }
        ))
        node_count += 1

        # Classes in file
        num_classes = np.random.randint(1, 4)
        for class_idx in range(num_classes):
            pattern = patterns[(file_idx + class_idx) % len(patterns)]
            class_name = f"{pattern[0].title().replace('_', '')}_{file_idx}_{class_idx}"
            class_id = f"class:{file_path}:{class_name}"

            graph.add_node(GraphNode(
                node_id=class_id,
                node_type=NodeType.CLASS,
                name=class_name,
                file_path=file_path,
                metadata={
                    "docstring": pattern[1],
                    "methods": ["__init__", "process", "validate"],
                    "bases": ["BaseHandler"]
                }
            ))
            graph.add_edge(GraphEdge(file_id, class_id, EdgeType.CONTAINS))
            node_count += 1

            # Functions in class
            num_funcs = np.random.randint(2, 6)
            for func_idx in range(num_funcs):
                func_name = f"{pattern[0]}_{func_idx}"
                func_id = f"func:{file_path}:{class_name}.{func_name}"

                graph.add_node(GraphNode(
                    node_id=func_id,
                    node_type=NodeType.FUNCTION,
                    name=func_name,
                    file_path=file_path,
                    metadata={
                        "signature": f"def {func_name}(self, *args, **kwargs)",
                        "docstring": f"{pattern[1]} - method {func_idx}",
                        "body": pattern[2]
                    }
                ))
                graph.add_edge(GraphEdge(class_id, func_id, EdgeType.CONTAINS))
                function_nodes.append(func_id)
                node_count += 1

    # Add call relationships (random subset)
    num_calls = min(len(function_nodes) * 2, 500)
    for _ in range(num_calls):
        src = np.random.choice(function_nodes)
        tgt = np.random.choice(function_nodes)
        if src != tgt and not graph.graph.has_edge(src, tgt):
            graph.add_edge(GraphEdge(src, tgt, EdgeType.CALLS, weight=np.random.uniform(0.5, 1.0)))

    print(f"Generated {graph.node_count} nodes, {graph.edge_count} edges")
    return graph


def benchmark_embedding_generation(graph: RepoGraph) -> dict:
    """Benchmark embedding generation speed."""
    print("\n--- Embedding Generation Benchmark ---")

    manager = GraphEmbeddingManager(graph, batch_size=32)

    # Time full embedding
    start = time.perf_counter()
    stats = manager.embed_all_nodes(force=True)
    elapsed = time.perf_counter() - start

    nodes_per_sec = stats["embedded"] / max(elapsed, 0.001)

    result = {
        "total_nodes": stats["embedded"] + stats["skipped"],
        "embedded": stats["embedded"],
        "errors": stats["errors"],
        "elapsed_seconds": round(elapsed, 2),
        "nodes_per_second": round(nodes_per_sec, 1),
    }

    print(f"  Embedded: {stats['embedded']} nodes")
    print(f"  Errors: {stats['errors']}")
    print(f"  Time: {elapsed:.2f}s ({nodes_per_sec:.1f} nodes/sec)")

    return result


def benchmark_semantic_search(graph: RepoGraph) -> dict:
    """Benchmark semantic search latency."""
    print("\n--- Semantic Search Benchmark ---")

    engine = SemanticSearchEngine(graph)

    # Build index
    start = time.perf_counter()
    indexed = engine.build_index()
    index_time = time.perf_counter() - start

    print(f"  Indexed: {indexed} nodes in {index_time:.3f}s")

    # Test queries
    queries = [
        "async HTTP request handler with error handling",
        "cache manager with expiration",
        "database query with connection pool",
        "websocket message processing",
        "rate limiting token bucket",
        "file upload with chunks",
        "authentication JWT token",
        "logging configuration setup",
        "YAML config loader",
        "data validation pipeline",
    ]

    latencies = []
    for query in queries:
        start = time.perf_counter()
        results = engine.search_semantic(query, top_k=10)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    avg_latency = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)

    result = {
        "index_time_seconds": round(index_time, 3),
        "indexed_nodes": indexed,
        "queries_tested": len(queries),
        "avg_latency_ms": round(avg_latency, 2),
        "p50_latency_ms": round(p50, 2),
        "p99_latency_ms": round(p99, 2),
    }

    print(f"  Queries: {len(queries)}")
    print(f"  Avg latency: {avg_latency:.2f}ms")
    print(f"  P50 latency: {p50:.2f}ms")
    print(f"  P99 latency: {p99:.2f}ms")

    return result


def benchmark_hybrid_search(graph: RepoGraph) -> dict:
    """Benchmark hybrid semantic + structural search."""
    print("\n--- Hybrid Search Benchmark ---")

    engine = SemanticSearchEngine(graph)
    engine.build_index()

    queries = [
        "async request handler",
        "cache with TTL",
        "database connection",
        "websocket handler",
        "authentication middleware",
    ]

    latencies = []
    for query in queries:
        start = time.perf_counter()
        results = engine.search_hybrid(query, top_k=10, expand_depth=2)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    avg_latency = np.mean(latencies)

    result = {
        "queries_tested": len(queries),
        "avg_latency_ms": round(avg_latency, 2),
        "includes_graph_expansion": True,
    }

    print(f"  Queries: {len(queries)}")
    print(f"  Avg latency (with graph expansion): {avg_latency:.2f}ms")

    return result


def benchmark_recall_accuracy(graph: RepoGraph) -> dict:
    """
    Measure recall accuracy using synthetic ground-truth queries.

    We know the patterns used in generation, so we can check
    if semantic search retrieves the correct pattern types.
    """
    print("\n--- Recall Accuracy Benchmark ---")

    engine = SemanticSearchEngine(graph)
    engine.build_index()

    # Ground-truth queries (pattern -> expected name substring)
    test_cases = [
        ("async HTTP handler with await", "async_handler"),
        ("data processing and validation loop", "data_processor"),
        ("LRU cache expiration management", "cache_manager"),
        ("SQL database query execution", "database_query"),
        ("JWT token authentication", "auth_middleware"),
        ("chunked file upload streaming", "file_uploader"),
        ("websocket message loop", "websocket_handler"),
        ("rate limit token bucket", "rate_limiter"),
        ("structured JSON logging", "logger_config"),
        ("YAML configuration loading", "config_loader"),
    ]

    hits = 0
    total = len(test_cases)

    for query, expected_pattern in test_cases:
        results = engine.search_semantic(query, top_k=5)

        # Check if any top-5 result contains the expected pattern
        found = any(expected_pattern in r.name.lower() for r in results)
        if found:
            hits += 1

    recall = hits / total if total > 0 else 0

    result = {
        "test_cases": total,
        "hits_at_5": hits,
        "recall_at_5": round(recall, 3),
    }

    print(f"  Test cases: {total}")
    print(f"  Hits@5: {hits}")
    print(f"  Recall@5: {recall:.1%}")

    return result


def measure_memory_usage() -> dict:
    """Measure current memory usage."""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            "rss_mb": round(mem_info.rss / 1024 / 1024, 1),
            "vms_mb": round(mem_info.vms / 1024 / 1024, 1),
        }
    except ImportError:
        return {"note": "psutil not installed, memory stats unavailable"}


def run_benchmark():
    """Run full benchmark suite."""
    print("=" * 60)
    print("SAGA Embedding Engine Benchmark")
    print("=" * 60)

    # Memory baseline
    mem_before = measure_memory_usage()
    print(f"\nMemory before: {mem_before}")

    # Generate mock repo
    graph = generate_mock_repo(num_files=100)

    # Run benchmarks
    embedding_stats = benchmark_embedding_generation(graph)
    search_stats = benchmark_semantic_search(graph)
    hybrid_stats = benchmark_hybrid_search(graph)
    recall_stats = benchmark_recall_accuracy(graph)

    # Memory after
    mem_after = measure_memory_usage()
    print(f"\nMemory after: {mem_after}")

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"""
Embedding Generation:
  - Nodes: {embedding_stats['embedded']}
  - Time: {embedding_stats['elapsed_seconds']}s
  - Throughput: {embedding_stats['nodes_per_second']} nodes/sec

Semantic Search:
  - Index time: {search_stats['index_time_seconds']}s
  - Avg latency: {search_stats['avg_latency_ms']}ms
  - P99 latency: {search_stats['p99_latency_ms']}ms

Hybrid Search:
  - Avg latency: {hybrid_stats['avg_latency_ms']}ms

Recall Accuracy:
  - Recall@5: {recall_stats['recall_at_5']:.1%} ({recall_stats['hits_at_5']}/{recall_stats['test_cases']})

Memory:
  - Before: {mem_before.get('rss_mb', 'N/A')} MB
  - After: {mem_after.get('rss_mb', 'N/A')} MB
""")

    return {
        "embedding": embedding_stats,
        "semantic_search": search_stats,
        "hybrid_search": hybrid_stats,
        "recall": recall_stats,
        "memory": {"before": mem_before, "after": mem_after}
    }


if __name__ == "__main__":
    run_benchmark()
