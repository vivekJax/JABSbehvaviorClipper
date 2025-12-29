#!/usr/bin/env python3
"""
Parse documentation registry and provide utilities for accessing documentation.

Usage:
    python3 docs/parse_docs.py --list
    python3 docs/parse_docs.py --find "getting-started"
    python3 docs/parse_docs.py --script extract_bout_features.py
"""

import yaml
import json
import argparse
import os
from pathlib import Path


def load_registry(registry_path='docs/documentation_registry.yaml'):
    """Load the documentation registry."""
    registry_file = Path(__file__).parent / registry_path
    if not registry_file.exists():
        registry_file = Path(registry_path)
    
    with open(registry_file, 'r') as f:
        return yaml.safe_load(f)


def list_docs(registry, category=None, audience=None, tag=None):
    """List all documentation matching filters."""
    docs = registry['documentation']
    
    if category:
        docs = [d for d in docs if d.get('category') == category]
    if audience:
        docs = [d for d in docs if audience in d.get('audience', [])]
    if tag:
        docs = [d for d in docs if tag in d.get('tags', [])]
    
    return docs


def find_script_docs(registry, script_name):
    """Find documentation for a specific script."""
    # Check Python scripts
    for script in registry['scripts'].get('python', []):
        if script['name'] == script_name:
            return script
    
    # Check R scripts
    for script in registry['scripts'].get('r', []):
        if script['name'] == script_name:
            return script
    
    return None


def get_doc_by_id(registry, doc_id):
    """Get documentation entry by ID."""
    for doc in registry['documentation']:
        if doc.get('id') == doc_id:
            return doc
    return None


def main():
    parser = argparse.ArgumentParser(description='Parse documentation registry')
    parser.add_argument('--list', action='store_true', help='List all documentation')
    parser.add_argument('--category', help='Filter by category')
    parser.add_argument('--audience', help='Filter by audience')
    parser.add_argument('--tag', help='Filter by tag')
    parser.add_argument('--find', help='Find documentation by ID or title')
    parser.add_argument('--script', help='Find documentation for a script')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--registry', default='docs/documentation_registry.yaml',
                       help='Path to registry file')
    
    args = parser.parse_args()
    
    registry = load_registry(args.registry)
    
    if args.script:
        script_doc = find_script_docs(registry, args.script)
        if script_doc:
            if args.json:
                print(json.dumps(script_doc, indent=2))
            else:
                print(f"Script: {script_doc['name']}")
                print(f"Description: {script_doc['description']}")
                print(f"Category: {script_doc['category']}")
                print(f"Documentation IDs: {', '.join(script_doc['documentation'])}")
        else:
            print(f"Script '{args.script}' not found")
        return
    
    if args.find:
        doc = get_doc_by_id(registry, args.find)
        if not doc:
            # Try to find by title
            docs = [d for d in registry['documentation'] 
                   if args.find.lower() in d.get('title', '').lower()]
            if docs:
                doc = docs[0]
        
        if doc:
            if args.json:
                print(json.dumps(doc, indent=2))
            else:
                print(f"Title: {doc['title']}")
                print(f"Path: {doc['path']}")
                print(f"Category: {doc['category']}")
                print(f"Audience: {', '.join(doc['audience'])}")
                print(f"Tags: {', '.join(doc['tags'])}")
                print(f"Description: {doc['description']}")
        else:
            print(f"Documentation '{args.find}' not found")
        return
    
    # List documentation
    docs = list_docs(registry, category=args.category, 
                    audience=args.audience, tag=args.tag)
    
    if args.json:
        print(json.dumps(docs, indent=2))
    else:
        print(f"Found {len(docs)} documentation entries:\n")
        for doc in docs:
            print(f"- {doc['title']} ({doc['id']})")
            print(f"  Path: {doc['path']}")
            print(f"  Category: {doc['category']}")
            print(f"  Tags: {', '.join(doc['tags'])}")
            print()


if __name__ == '__main__':
    main()

