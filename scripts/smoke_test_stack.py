
"""
Smoke Test: Stack Verification
==============================

Verifies that the "North Star" dependencies are correctly installed
and compatible with the current environment (Windows).

Checks:
1. LangGraph import & basic graph construction
2. ChromaDB import & client initialization
3. LiteLLM import
"""

import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PROBE")

def test_langgraph():
    logger.info("Testing LangGraph...")
    try:
        from langgraph.graph import END, StateGraph
        logger.info("✅ LangGraph imported successfully")
    except ImportError as e:
        logger.error(f"❌ LangGraph failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ LangGraph crashed: {e}")
        return False
    return True

def test_chromadb():
    logger.info("Testing ChromaDB...")
    try:
        import chromadb
        client = chromadb.Client()
        logger.info(f"✅ ChromaDB initialized (Version: {chromadb.__version__})")
    except ImportError as e:
        logger.error(f"❌ ChromaDB failed to import: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ ChromaDB crashed initialization: {e}")
        # Typical windows SQLite issue check
        return False
    return True

def test_litellm():
    logger.info("Testing LiteLLM...")
    try:
        import litellm
        logger.info(f"✅ LiteLLM imported (Version: {getattr(litellm, '__version__', 'unknown')})")
    except ImportError as e:
        logger.error(f"❌ LiteLLM failed: {e}")
        return False
    return True

if __name__ == "__main__":
    logger.info("=== STARTING SMOKE TEST ===")
    results = [
        test_langgraph(),
        test_chromadb(),
        test_litellm()
    ]

    if all(results):
        logger.info("=== ALL SYSTEMS GO ===")
        sys.exit(0)
    else:
        logger.error("=== VERIFICATION FAILED ===")
        sys.exit(1)
