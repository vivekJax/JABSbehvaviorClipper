---
title: "Documentation Guide"
description: "Guide for maintaining and using the documentation system"
category: "reference"
audience: ["developers", "maintainers"]
tags: ["documentation", "maintenance", "guide"]
---

# Documentation Guide

This guide explains how to maintain and use the machine-readable documentation system.

## Structure

The documentation system consists of:

1. **YAML Frontmatter**: Each markdown file has YAML frontmatter with metadata
2. **Documentation Registry**: Central YAML file indexing all documentation
3. **Parser Script**: Python utility to query and parse documentation

## Adding New Documentation

### Step 1: Add YAML Frontmatter

Every markdown file should start with YAML frontmatter:

```yaml
---
title: "Document Title"
description: "Brief description"
category: "user-guide"  # or "technical", "reference", "overview"
audience: ["users", "developers"]  # or ["analysts"], ["researchers"]
tags: ["tag1", "tag2"]
documentation_id: "unique-id"
---
```

### Step 2: Register in documentation_registry.yaml

Add an entry to `docs/documentation_registry.yaml`:

```yaml
documentation:
  - id: "unique-id"
    title: "Document Title"
    path: "../path/to/file.md"
    category: "user-guide"
    audience: ["users"]
    tags: ["tag1", "tag2"]
    description: "Brief description"
```

### Step 3: Link Scripts (if applicable)

If the documentation relates to a script, add it to the script's documentation list:

```yaml
scripts:
  python:
    - name: "script_name.py"
      documentation: ["doc-id-1", "doc-id-2"]
```

## Using the Parser

### List All Documentation

```bash
python3 docs/parse_docs.py --list
```

### Filter by Category

```bash
python3 docs/parse_docs.py --list --category user-guide
```

### Filter by Tag

```bash
python3 docs/parse_docs.py --list --tag "getting-started"
```

### Find Documentation for a Script

```bash
python3 docs/parse_docs.py --script extract_bout_features.py
```

### Find by ID

```bash
python3 docs/parse_docs.py --find "quick-start"
```

### JSON Output

```bash
python3 docs/parse_docs.py --list --json
```

## Categories

- **overview**: Main documentation and project overview
- **user-guide**: Step-by-step guides for users
- **technical**: Implementation details and technical documentation
- **reference**: API references and indexes

## Audience

- **users**: End users running the pipeline
- **developers**: Developers modifying the code
- **analysts**: Data analysts using the tools
- **researchers**: Researchers understanding the methodology

## Tags

Use consistent tags:
- `getting-started`: Quick start guides
- `pipeline`: Pipeline workflow documentation
- `statistics`: Statistical methodology
- `performance`: Performance optimization
- `troubleshooting`: Debugging and FAQ
- `video-processing`: Video-related documentation
- `clustering`: Clustering methods
- `outliers`: Outlier detection

## Best Practices

1. **Keep frontmatter consistent**: Use the same structure across all files
2. **Use descriptive IDs**: Documentation IDs should be kebab-case and descriptive
3. **Tag appropriately**: Use 2-5 relevant tags per document
4. **Update registry**: Always update the registry when adding/removing docs
5. **Link scripts**: Link documentation to relevant scripts
6. **Keep descriptions concise**: 1-2 sentences maximum

## Validation

Validate the registry:

```bash
python3 -c "import yaml; yaml.safe_load(open('docs/documentation_registry.yaml'))"
```

## Examples

### Example: Adding User Guide

1. Create `docs/user-guides/new-feature.md` with frontmatter
2. Add entry to registry under `documentation:`
3. Link to relevant scripts if applicable

### Example: Adding Technical Doc

1. Create `docs/technical/implementation-details.md` with frontmatter
2. Add entry to registry
3. Tag with appropriate technical tags

## Maintenance

- Review registry quarterly for outdated entries
- Ensure all markdown files have frontmatter
- Keep tags consistent across similar documents
- Update last_updated dates when modifying docs

