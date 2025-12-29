---
title: "Documentation Index"
description: "Machine-readable documentation registry and navigation"
category: "reference"
audience: ["users", "developers"]
tags: ["documentation", "index", "reference"]
---

# Documentation

This directory contains the documentation registry and organized documentation files.

## Documentation Registry

The [documentation_registry.yaml](documentation_registry.yaml) file provides a machine-readable index of all documentation in the project. It includes:

- **Metadata**: Version, last updated date, maintainer
- **Documentation entries**: Title, path, category, audience, tags, description
- **Script references**: Links between scripts and their documentation
- **Output structure**: Description of output directories and files

## Using the Registry

### Python Example

```python
import yaml

with open('docs/documentation_registry.yaml', 'r') as f:
    registry = yaml.safe_load(f)

# Find all user guides
user_guides = [doc for doc in registry['documentation'] 
               if doc['category'] == 'user-guide']

# Find documentation for a script
script_docs = [doc for doc in registry['scripts']['python']
               if doc['name'] == 'extract_bout_features.py']
```

### Command Line

```bash
# List all documentation files
yq '.documentation[].path' docs/documentation_registry.yaml

# Find documentation by tag
yq '.documentation[] | select(.tags[] == "getting-started")' docs/documentation_registry.yaml

# Get script documentation
yq '.scripts.python[] | select(.name == "run_complete_analysis.py")' docs/documentation_registry.yaml
```

## Documentation Categories

- **overview**: Main documentation and project overview
- **user-guide**: Step-by-step guides for users
- **technical**: Implementation details and technical documentation
- **reference**: API references and indexes

## Audience Tags

- **users**: End users running the pipeline
- **developers**: Developers modifying the code
- **analysts**: Data analysts using the tools
- **researchers**: Researchers understanding the methodology

## Tags

Common tags include:
- `getting-started`: Quick start guides
- `pipeline`: Pipeline workflow documentation
- `statistics`: Statistical methodology
- `performance`: Performance optimization
- `troubleshooting`: Debugging and FAQ

## Structure

```
docs/
├── documentation_registry.yaml  # Machine-readable index
├── README.md                    # This file
├── user-guides/                 # User-facing documentation
├── technical/                   # Technical documentation
└── api-reference/              # API documentation
```

