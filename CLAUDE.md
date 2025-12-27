# CLAUDE.md - AI Agent Development Guide

## IMPORTANT - Cost Optimization

**Every call to the AI model API incurs a cost and requires electricity. Be smart and make as few requests as possible. Each request gets subsequently more expensive as the context increases.**

### Efficient Development Practices

1. **Monitor context usage** - Keep under 40% (80K/200K tokens) when possible
   - Check regularly with token counter
   - Prepare NEXT_SESSION_PROMPT.md as context approaches 40%
   - Summarize, compact & commit work to reset context (reference filepaths and line numbers)
   - Use offset/limit when reading large files

2. **Add learnings with references** to avoid repeating mistakes
   - Document fixes with file:line references
   - Example: `src/module/file.py:42 (--flag-name)`
   - Update CLAUDE.md or relevant docs with specific locations

3. **Prefer command line tools** to reduce context usage
   - Use `jq` for JSON, `grep` for search, `git` for history
   - Avoid reading entire files when possible
   - Never launch interactive versions of CLI tools that wait for input (use GIT_TERMINAL_PROMPT=0, --no-interactive, etc.)
   - **Prefer language server plugins over grep** when available (e.g., `pylsp`, `pyright`, `lsp-pyright`) for go-to-definition, find-references, and symbol search - more accurate and context-aware than text search

4. **Ignore line length and type errors until the very end** - use command line tools to fix those (black, flake8, ruff)

### Think Holistically

Before diving into code:
- What is the **PURPOSE** of this tool?
- Why is it failing on this issue?
- Is this a symptom of a **larger architectural problem**?
- Take a step back and analyze the **root cause**

Don't just fix symptoms. Understand the underlying architecture and design decisions before making changes.

### Development Best Practices

- **Commit often** - Small, focused commits are easier to review and rollback
- **Use `git add -p` for staging** - Interactive staging to add only relevant changes
  - Shows each change and lets you choose what to stage
  - Creates logical commits with related changes only
  - Prevents accidentally committing unrelated work
- **Use TDD whenever possible** - Write tests first, then implement
- **Keep it simple** - Always think hard and evaluate more complex approaches and alternative approaches before moving forward
- **Measure twice, cut once** - Plan before implementing

### Using Subagents

Use subagents to reduce context usage and parallelize work. Different models are suited for different tasks:

**Research Subagent (Sonnet)**
- Use for searching the repo, web research, and gathering context
- Should report back with file paths, line numbers, and relevant excerpts
- Good for: "Find all usages of X", "How does Y work in this codebase", "What's the current best practice for Z"

**Implementation Subagent (Haiku)**
- Use for executing small, well-defined chunks of work
- Instruct to complete one small task, then report back for review before continuing
- Good for: "Add this field to the model", "Write this test case", "Update these config values"
- Keep scope narrow - easier to review and rollback

**When to use subagents:**
- Complex multi-file changes that benefit from research-then-implement pattern
- When context is getting large and you need focused work
- When parallelizing independent tasks

**Pattern:**
1. Research subagent gathers context and reports findings
2. Main agent reviews and plans implementation steps
3. Implementation subagent executes one step at a time
4. Main agent reviews each step before proceeding

---

## Documentation Organization

### Stay Organized

- ❌ NO markdown files in `/` (README.md and CLAUDE.md only)
- ✅ ALL new docs go under `/docs/claude/` or similar organized structure
- ✅ Always include file paths and line numbers in documentation

### Creating New Documentation

All new documentation created during AI sessions MUST go under `docs/claude/`:

```
docs/claude/
├── ./               - Main reference docs
├── sessions/        - Session summaries, notes to keep during a session
├── archive/         - Historical/completed work
├── planning/        - Planning docs, re-review findings
├── prompts/         - Session prompts
└── debugging/       - Production debugging case studies
```

**Purpose of docs/claude/**: These docs are for **AI agent acceleration**, not human explanation. They focus on:
- **Efficient context usage** through excellent, accurate compaction
- File paths and line numbers for quick navigation
- Gotchas, learnings, and corrections
- Reducing redundant exploration in future sessions

**Key principles:**
- One doc = one purpose (don't mix session notes with reference material)
- Always include file:line references
- Update existing docs rather than creating duplicates
- Archive session-specific docs when work completes

### Using the Documentation Index

**`docs/claude/INDEX.md`** is a searchable index & quick reference of all documentation in the repository with a compacted description of the utility of each file.

**When to use it:**
- Before starting work on a feature/bug, search for relevant keywords
- Find related documentation without reading every file
- Identify which docs are session notes vs. permanent reference

**How to search efficiently:**
```bash
# Search for keywords in the index
grep -i "keyword.*pattern" docs/claude/INDEX.md
grep -i "term1" docs/claude/INDEX.md | grep -i "term2"
```

Search git commit history if nothing found. Search repo if nothing found.

**Each file in INDEX.md should include:**
- Relative file path
- Date last indexed
- Compacted description of most helpful & unique content
- Keywords and key concepts for searching & recall
- List of most relevant source code files, functions & line numbers
- File creation dates and authors (AI agent or human maintainer)
- Relevant git history & commits
- Links between related docs

---

## Command Line Shortcuts

### Use git for quick file inspection

```bash
# Count files
git ls-files | wc -l

# Find files by pattern
git ls-files | grep "*.sql"

# Check recent changes
git log --oneline -10
```

### Use jq for JSON inspection

```bash
# Parse field
echo '{"key":"value"}' | jq .key

# Pretty print JSON
cat file.json | jq .
```

---

# PROJECT-SPECIFIC SECTIONS
# (Fill these in for each project)

---

## Project Overview

<!-- Brief description of the project -->

### Quick Stats

<!-- Key metrics, counts, coverage -->

### Key Technologies

<!-- Languages, frameworks, databases, external services -->

---

## Architecture Overview

<!-- 
- Data flow diagram (ASCII or description)
- Key design patterns used
- Request flow
-->

### Module Structure

```
project-root/
├── ...
└── ...
```

---

## Key Files to Understand

<!--
List the critical files with brief descriptions:
1. `/path/to/main/app.py` - Main application entry point
2. `/path/to/config.py` - Configuration settings
3. ...
-->

---

## Development Workflow

### Setup

```bash
# Setup commands go here
```

### Common Commands

```bash
# Build, test, run commands
```

---

## Testing Strategy

<!--
- Where tests live
- How to run tests
- Coverage expectations
- Test data approach
-->

---

## Third-Party Integrations

<!--
List external services and their purpose:
- **Service Name**: Purpose
-->

---

## Learnings and Gotchas

*(Add items here as you discover them during development)*

### Database

-

### API

-

### Performance

-

### Testing

-

### Documentation Quality

<!-- 
IMPORTANT: Note any docs that are aspirational vs verified.
Always verify AI-written documentation against actual code before trusting.
-->

---

## Common Tasks

### Task 1: [Name]

```bash
# Steps
```

### Task 2: [Name]

```bash
# Steps
```

---

## Detailed Documentation

For more detailed information, see:
- `docs/claude/architecture.md` - Detailed architecture
- `docs/claude/development.md` - Development workflow
- `docs/claude/troubleshooting.md` - Common issues and solutions

---

*Last updated: YYYY-MM-DD*
