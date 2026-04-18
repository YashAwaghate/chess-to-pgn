# gen-tests

Generate a pytest test file for a given source module in this chess-to-pgn project.

## Usage

```
/gen-tests src/pipeline/fen_generator.py
```

## What this skill does

1. Reads the target source file
2. Identifies every public function / class (those without a leading `_`)
3. Drafts a `tests/test_<module_name>.py` file that:
   - Imports the module under test with the full `src.*` path
   - Groups test cases under a class per public function
   - Covers: happy path, edge cases (empty input, boundary values), and error paths
   - Uses `pytest` idioms: `pytest.raises`, `tmp_path`, parametrize where appropriate
   - Adds a module-level docstring listing coverage targets

## Prompting rules applied here

- **Privacy**: no real game data, credentials, or S3 paths are included in prompts — only type signatures and doc-strings are sent
- **Context first**: system prompt explains the module's role in the pipeline before asking for tests
- **Narrow scope**: only the functions in the target file are tested; transitive dependencies are mocked or skipped
- **Sample I/O provided**: at least one concrete input → expected output pair per function
- **Output format fixed**: always `pytest`, always grouped by class, always uses the `tests/` directory

## Template prompt sent to the model

```
You are writing pytest unit tests for the chess-to-pgn project.

Module: $FILE_PATH
Role in pipeline: $ONE_LINE_ROLE    # e.g. "converts 64-square predictions to a FEN string"

Public API (signatures only — no implementation):
$SIGNATURES

Coverage targets:
- $FUNCTION_1: happy path, empty input, boundary
- $FUNCTION_2: ...

Sample input → expected output:
$EXAMPLES

Constraints:
- Use pytest; no unittest
- One TestClass per public function
- No network calls, no S3, no filesystem (use tmp_path for file tests)
- Do not import private helpers (_prefixed); test them through the public API

Output: a complete, runnable tests/test_$MODULE_NAME.py file.
```
