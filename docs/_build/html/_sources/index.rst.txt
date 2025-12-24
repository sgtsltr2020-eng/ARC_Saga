SAGA Documentation
==================

**Systematized Autonomous Generative Assistant**

SAGA is an advanced multi-agent system for building world-class software.
It combines:

* **The Warden** - Delegation and quality enforcement
* **Mimiry** - The immutable oracle of software truth
* **SagaCodex** - FAANG-level engineering standards
* **SagaConstitution** - 15 meta-rules governing system behavior

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   architecture
   api/modules
   testing
   architecture

Introduction
============

SAGA is built on a hierarchical delegation model:

1. **User** provides natural language intent
2. **SAGA** interprets and creates structured directives
3. **The Warden** decomposes tasks and spawns coding agents
4. **Mimiry** (Oracle) is consulted when discrepancies arise
5. **SagaCodex** provides the ideal standards to measure against

Philosophy
----------

**Mimiry + SagaCodex = Uncompromising Pursuit of Perfection**

* **Mimiry**: The immutable oracle (the North Star)
* **LoreBook**: Practical wisdom from experience (the map)
* **The Warden**: Enforcer of quality and delegator
* **Coding Agents**: Transient specialists that execute tasks

Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install -e .
   pytest tests/ --cov=saga --cov-report=term-missing

Basic Usage
-----------

.. code-block:: python

   from saga.core.warden import Warden
   from saga.core.mimiry import Mimiry

   # Initialize
   warden = Warden()
   mimiry = Mimiry()

   # Consult Oracle
   response = await mimiry.consult_on_discrepancy(
       question="Should functions have type hints?",
       context={"language": "python"}
   )

   print(response.canonical_answer)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
