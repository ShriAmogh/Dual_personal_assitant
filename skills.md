# Development Guidelines and Skills

This document outlines the standard operating procedures and technical requirements for contributing to this project. All contributors must adhere to these practices to ensure codebase stability and professional standards.

## 1. Environment Management
* **Virtual Environments:** Always use a virtual environment for Python development. The standard directory name for this project is `.venv`.
* **Setup:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```
* **Isolation:** Never install packages globally. Ensure the `.venv` is added to your `.gitignore`.

## 2. Implementation Protocol
Before writing a single line of production code, an **Implementation Plan** must be created and reviewed.
* **Phase 1: Analysis:** Understand the requirements and identify the touchpoints in the existing codebase.
* **Phase 2: Design:** Outline the logic, API changes, and data structures.
* **Phase 3: Validation:** Confirm the plan covers edge cases and security considerations.

## 3. Testing & Refactoring Strategy
Stability is prioritized over speed. Follow the "Test-First" workflow:
1.  **Baseline Testing:** Before refactoring existing code or adding new features, run the existing test suite to ensure the current state is stable.
2.  **New Code:** Write unit or integration tests for the new functionality.
3.  **Refactoring:** Only refactor code once there is a passing test suite to act as a safety net. If a refactor breaks a test, fix the logic immediately.

## 4. Codebase Structure
The project follows a modular, clean architecture.
* **Modularity:** Logic should be separated by domain (e.g., `services/`, `models/`, `api/`).
* **Clarity:** Use descriptive naming conventions and type hinting for all Python functions.
* **Standard Layout:**
    ```text
    project_root/
    ├── app/            # Source code
    ├── tests/          # Test suite
    ├── docs/           # Documentation
    ├── .venv/          # Local environment (ignored)
    └── requirements.txt
    ```

## 5. Grill Me (Requirement Clarification)
If a task, ticket, or feature request is ambiguous, **do not guess.**
* **The "Grill Me" Rule:** The developer is encouraged to ask as many clarifying questions as necessary until the intent is 100% clear.
* **Criteria for Clarity:**
    * What is the expected input/output?
    * What are the edge cases?
    * Are there specific performance or security constraints?
    * How does this impact existing features?
* **Action:** If the task is unclear, stop and "grill" the stakeholder/maintainer with a list of targeted questions before proceeding to the Implementation Plan.