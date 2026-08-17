"""
==============================================================================
playbook.py
==============================================================================

This file contains functions for parsing and manipulating the playbook.

"""
import json
import re
from utils import get_section_slug, repair_json_escapes

def normalize_section_name(name):
    """
    Canonical key for a section, from either a display header or a curator's section field.

    The hyphen replacement is what makes "## PROBLEM-SOLVING HEURISTICS" and the
    curator's "problem_solving_heuristics" resolve to the same key. Without it the
    header normalized to "problem-solving_heuristics", never matched, and every
    bullet the curator filed under its most-used section was rerouted to OTHERS.
    """
    return name.lower().strip().replace(' ', '_').replace('&', 'and').replace('-', '_')


def parse_playbook_line(line):
    """Parse a single playbook line to extract components"""
    # Pattern: [id] helpful=X harmful=Y :: content
    pattern = r'\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)'
    match = re.match(pattern, line.strip())
    
    if match:
        return {
            'id': match.group(1),
            'helpful': int(match.group(2)),
            'harmful': int(match.group(3)),
            'content': match.group(4),
            'raw_line': line
        }
    return None

def rendered_bullet_ids(playbook_text):
    """Ids present in a rendered playbook, in render order.

    The co-render denominator for co-citation: which rules retrieval actually put in front of
    the model on this call. Parsed from the rendering rather than asked of the memory, because
    the rendering is what the model saw -- a budget trim or a superseded-node filter applied
    after selection would otherwise go unrecorded.
    """
    ids = []
    for line in (playbook_text or '').split('\n'):
        parsed = parse_playbook_line(line)
        if parsed:
            ids.append(parsed['id'])
    return ids


def get_next_global_id(playbook_text):
    """Extract highest global ID and return next one"""
    max_id = 0
    lines = playbook_text.strip().split('\n')
    
    for line in lines:
        parsed = parse_playbook_line(line)
        if parsed:
            # Extract numeric part from ID
            id_match = re.search(r'-(\d+)$', parsed['id'])
            if id_match:
                num = int(id_match.group(1))
                max_id = max(max_id, num)
    
    return max_id + 1


def format_playbook_line(bullet_id, helpful, harmful, content):
    """Format a bullet into playbook line format"""
    return f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {content}"

def update_bullet_counts(playbook_text, bullet_tags):
    """Update helpful/harmful counts based on tags (Counter layer)"""
    lines = playbook_text.strip().split('\n')
    updated_lines = []
    
    # Create tag lookup - handle both old and new formats
    tag_map = {}
    if isinstance(bullet_tags, list) and len(bullet_tags) > 0:
        for tag in bullet_tags:
            if isinstance(tag, dict):
                # Handle both 'id' and 'bullet' keys for backwards compatibility
                bullet_id = tag.get('id') or tag.get('bullet', '')
                tag_value = tag.get('tag', 'neutral')
                if bullet_id:
                    tag_map[bullet_id] = tag_value
    
    if not tag_map:
        print("Warning: No valid bullet tags found to update counts")
        return playbook_text
    
    matched_ids = set()

    for line in lines:
        if line.strip().startswith('#') or not line.strip():
            # Preserve section headers and empty lines
            updated_lines.append(line)
            continue

        parsed = parse_playbook_line(line)
        if parsed and parsed['id'] in tag_map:
            matched_ids.add(parsed['id'])
            tag = tag_map[parsed['id']]
            if tag == 'helpful':
                parsed['helpful'] += 1
            elif tag == 'harmful':
                parsed['harmful'] += 1
            # neutral: no change
            
            # Reconstruct line with updated counts
            new_line = format_playbook_line(
                parsed['id'], parsed['helpful'], parsed['harmful'], parsed['content']
            )
            updated_lines.append(new_line)
        else:
            updated_lines.append(line)

    # Unmatched ids used to vanish silently, which made a stream of malformed tags
    # indistinguishable from a working counter update.
    unmatched = sorted(set(tag_map) - matched_ids)
    if unmatched:
        preview = ', '.join(unmatched[:5])
        suffix = ', ...' if len(unmatched) > 5 else ''
        print(f"Warning: {len(unmatched)} tagged id(s) not found in playbook: {preview}{suffix}")

    return '\n'.join(updated_lines)


# A rendered playbook line pasted back into the `content` of an ADD. The curator is shown the
# playbook as "[calc-00013] helpful=5 harmful=0 :: <rule>" and both prompts tell it not to
# repeat the id -- gemma4:e2b does it anyway, at a rate that scales with playbook size (0 of 221
# ADDs across every MBPP run, 5/15 on formula probe 3, 43/75 on the text arm, 34/65 on the graph
# arm). The counters are a snapshot of a *different* rule's track record, so they go into the
# generator prompt as rule text and never update again.
#
# Three forms observed, hence the optional groups: full counters, `helpful=` alone, and a bare
# id with no counters and no `::`.
#
# The id shape used to be pinned to `\w+-\d{5}`, i.e. what get_section_slug + the %05d in
# apply_curator_operations can actually produce. That assumption was wrong and leaked: the model
# does not paste the id it was shown, it *invents* one. Measured on the nine pre-registered
# formula runs, 10/15/26 (text) and 16/3/12 (graph) rules reached the final playbook with a
# prefix still attached, under ids the harness cannot emit -- `[calc-00047_new]`,
# `[calc-00018_priority]`, `[calc-0016_new]` (four digits), `[ggm-r]` (no digits at all).
# The rates are similar in both arms, so this never biased a comparison; it just put a frozen
# snapshot of some other rule's counters into the generator prompt as rule text.
#
# Two tiers, because the evidence differs. The counters or a ` :: ` separator make a paste
# unambiguous, so the bracket may hold anything short. A *bare* bracket is only stripped when it
# is id-shaped -- no whitespace, contains a hyphen, at least four characters -- because a rule
# may legitimately open with a bracketed clause and one text-arm rule is an entire formula
# wrapped in brackets ("[IRP Theorem for Forward Rates: To calculate ...]"), which the looser
# tier would delete outright.
_COUNTERS = r'(?:\s*helpful\s*=\s*\d+)?(?:\s*harmful\s*=\s*\d+)?'
_RENDERED_PREFIX = re.compile(
    r'^(?:'
    r'\[[^\[\]]{1,40}\]' + _COUNTERS + r'\s*::\s*'          # any short tag, when `::` follows
    r'|\[[^\[\]]{1,40}\]\s*helpful\s*=\s*\d+' + _COUNTERS + r'\s*(?:::\s*)?'   # ... or counters
    r'|\[(?=[^\[\]\s]*-)[^\[\]\s]{4,}\]\s*'                 # bare, id-shaped only
    r')'
)


def strip_rendered_prefix(content):
    """Remove pasted rendered-bullet prefixes from a curator's content field.

    Returns (cleaned_content, n_prefixes_removed). Applied repeatedly because the paste
    nests: one text-arm ADD arrived as
    "[err-00017] helpful=4 harmful=0 :: [err-00011] helpful=3 harmful=0 :: When faced with...".
    """
    text = (content or '').strip()
    removed = 0
    while True:
        match = _RENDERED_PREFIX.match(text)
        if not match or match.end() == 0:
            return text, removed
        text = text[match.end():].strip()
        removed += 1


def _dedup_key(content):
    """Canonical form of a rule's text, for exact-match duplicate detection.

    Exact match after normalisation only -- deliberately no fuzzy or embedding similarity.
    Semantic near-duplicate detection is a GraphMemory capability; putting it in the shared
    write path would give the flat playbook a second treatment and make any arm difference
    unattributable. Near-duplicates surviving is by design: collapsing them is what `refines`
    is for, and that is the thing under test.
    """
    text, _ = strip_rendered_prefix(content)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.casefold().rstrip(' .;:,')


def sanitize_curator_operations(operations, current_playbook):
    """
    Enforce, in code, two things the curator prompt already asks for and the model ignores.

    1. Strip rendered-bullet prefixes pasted into `content` (see _RENDERED_PREFIX).
    2. Drop an ADD whose text already exists in the playbook the curator was just shown,
       or that repeats an earlier ADD in the same batch. 23 of the text arm's 75 ADDs were
       verbatim re-adds; an ADD-only curator has no way to notice, so every one of them
       bought a fresh id, fresh zeroed counters and a permanent second copy in the prompt.

    Shared code, applied identically in all three arms -- the same pattern as
    Reflector._filter_bullet_tags. It is enforcement of the authors' instructions, not a
    change to them: the prompts are untouched.

    Note for the graph arm: render_for_curator() excludes superseded nodes, so a withdrawn
    rule is not in the dedup set and may legitimately be re-added. That is correct. The rule
    applied is "do not add what is in front of you", identically in both arms; the visible
    set differing between them *is* the treatment.

    Non-ADD operations pass through untouched.

    Returns (kept_operations, report).
    """
    existing = set()
    for line in (current_playbook or '').split('\n'):
        parsed = parse_playbook_line(line)
        if parsed:
            key = _dedup_key(parsed['content'])
            if key:
                existing.add(key)

    report = {'adds': 0, 'stripped': 0, 'emptied': 0, 'deduped': 0, 'kept': 0}
    seen_in_batch = set()
    kept = []

    for op in operations:
        if not isinstance(op, dict) or op.get('type') != 'ADD':
            kept.append(op)
            continue

        report['adds'] += 1
        original = op.get('content', '') or ''
        cleaned, removed = strip_rendered_prefix(original)
        if removed:
            report['stripped'] += 1

        if not cleaned:
            # The whole content was a pasted id. Nothing to add.
            report['emptied'] += 1
            continue

        key = _dedup_key(cleaned)
        if key in existing or key in seen_in_batch:
            report['deduped'] += 1
            continue

        seen_in_batch.add(key)
        if cleaned != original:
            op = dict(op)
            op['content'] = cleaned
        kept.append(op)
        report['kept'] += 1

    report['dropped'] = report['emptied'] + report['deduped']
    return kept, report


def apply_curator_operations(playbook_text, operations, next_id):
    """
    Apply curator operations to playbook
    
    TODO: Future Operations (not implemented yet)
    - UPDATE: Rewrite existing bullets to be more accurate or comprehensive
    - MERGE: Combine related bullets into stronger ones  
    - CREATE_META: Add high-level strategy sections
    - DELETE: Remove outdated or incorrect bullets (if needed)
    """
    lines = playbook_text.strip().split('\n')
    
    # Build section map
    sections = {}
    current_section = "general"
    section_line_map = {}  # Track which line each section header is on
    
    for i, line in enumerate(lines):
        if line.strip().startswith('##'):
            # Extract section name and normalize it
            section_header = line.strip()[2:].strip()
            current_section = normalize_section_name(section_header)
            section_line_map[current_section] = i
            if current_section not in sections:
                sections[current_section] = []
        elif line.strip():
            sections[current_section].append((i, line))
    
    # Process operations
    bullets_to_add = []
    
    for op in operations:
        op_type = op['type']
        
        # TODO: Future operation types (not implemented yet)
        # elif op_type == 'UPDATE':
        #     bullet_id = op.get('bullet_id', '')
                    #     new_content = op.get('content', '')
            #     bullets_to_update[bullet_id] = new_content
        # elif op_type == 'MERGE':
        #     source_ids = op.get('source_ids', [])
        #     bullets_to_delete.update(source_ids)
        #     # Add merged bullet logic here
        # elif op_type == 'CREATE_META':
        #     section_name = op.get('section_name', 'META_STRATEGIES')
        #     # Add meta section creation logic here
        
        if op_type == 'ADD':
            # Normalize section name from operation
            section_raw = op.get('section', 'general')
            section = normalize_section_name(section_raw)
            
            # Check if section exists, if not use 'others'
            if section not in sections and section != 'general':
                print(f"Warning: Section '{section_raw}' not found, adding to OTHERS")
                section = 'others'
            
            slug = get_section_slug(section)
            new_id = f"{slug}-{next_id:05d}"
            next_id += 1
            
            content = op.get('content', '')
            
            new_line = format_playbook_line(new_id, 0, 0, content)
            bullets_to_add.append((section, new_line))
            print(f"  Added bullet {new_id} to section {section}")
            

    
    # Rebuild playbook
    new_lines = []
    for line in lines:
        parsed = parse_playbook_line(line)
        if parsed:
            new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Add new bullets to appropriate sections
    final_lines = []
    current_section = None
    
    for line in new_lines:
        if line.strip().startswith('##'):
            # Before moving to new section, add any bullets for current section
            if current_section:
                section_adds = [b for s, b in bullets_to_add if s == current_section]
                final_lines.extend(section_adds)
                # Clear added bullets
                bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]
            
            section_header = line.strip()[2:].strip()
            current_section = normalize_section_name(section_header)
        final_lines.append(line)
    
    # Add remaining bullets to current section
    if current_section:
        section_adds = [b for s, b in bullets_to_add if s == current_section]
        final_lines.extend(section_adds)
        bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]
    
    # If there are still bullets to add (for sections that don't exist), add them to OTHERS
    if bullets_to_add:
        print(f"Warning: {len(bullets_to_add)} bullets have no matching section, adding to OTHERS")
        others_bullets = [b for s, b in bullets_to_add]
        # Find OTHERS section
        others_idx = -1
        for i, line in enumerate(final_lines):
            if line.strip() == "## OTHERS":
                others_idx = i
                break
        
        if others_idx >= 0:
            # Insert after OTHERS header
            for i, bullet in enumerate(others_bullets):
                final_lines.insert(others_idx + 1 + i, bullet)
        else:
            # Append to end
            final_lines.extend(others_bullets)
    
    return '\n'.join(final_lines), next_id

def get_playbook_stats(playbook_text):
    """Generate statistics about the playbook"""
    lines = playbook_text.strip().split('\n')
    stats = {
        'total_bullets': 0,
        'high_performing': 0,  # helpful > 5, harmful < 2
        'problematic': 0,      # harmful >= helpful
        'unused': 0,           # helpful + harmful = 0
        'by_section': {}
    }
    
    current_section = 'general'
    
    for line in lines:
        if line.strip().startswith('##'):
            current_section = line.strip()[2:].strip()
            continue
            
        parsed = parse_playbook_line(line)
        if parsed:
            stats['total_bullets'] += 1
            
            if parsed['helpful'] > 5 and parsed['harmful'] < 2:
                stats['high_performing'] += 1
            elif parsed['harmful'] >= parsed['helpful'] and parsed['harmful'] > 0:
                stats['problematic'] += 1
            elif parsed['helpful'] + parsed['harmful'] == 0:
                stats['unused'] += 1
            
            if current_section not in stats['by_section']:
                stats['by_section'][current_section] = {'count': 0, 'helpful': 0, 'harmful': 0}
            
            stats['by_section'][current_section]['count'] += 1
            stats['by_section'][current_section]['helpful'] += parsed['helpful']
            stats['by_section'][current_section]['harmful'] += parsed['harmful']
    
    return stats

def extract_json_from_text(text, json_key=None, _repaired=False):
    """Extract JSON object from text, handling various formats

    _repaired is internal: when every parse below has failed, the text is retried once with
    unrecognised backslash escapes doubled. On a numeric task the models write their working
    in LaTeX inside a string field -- ``\\sigma_d``, ``\\%``, ``\\text{...}`` -- which is not
    legal JSON escaping, so a schema-perfect response is rejected whole. It hit 24 of 50
    reflector calls on the formula probe. The retry runs only after a strict parse has
    already failed, so it can never change how a well-formed response is read.
    """
    try:
        # First, try to parse the entire response as JSON (JSON mode)
        try:
            result = json.loads(text.strip())
            return result
        except json.JSONDecodeError:
            pass
        
        # Fallback: Look for ```json blocks
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Try each match until we find valid JSON
            for match in matches:
                try:
                    json_str = match.strip()
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    continue
        
        # Improved JSON extraction using balanced brace counting
        # This handles deeply nested structures better
        def find_json_objects(text):
            """Find JSON objects using balanced brace counting"""
            json_objects = []
            i = 0
            while i < len(text):
                if text[i] == '{':
                    # Found start of potential JSON object
                    brace_count = 1
                    start = i
                    i += 1
                    
                    while i < len(text) and brace_count > 0:
                        if text[i] == '{':
                            brace_count += 1
                        elif text[i] == '}':
                            brace_count -= 1
                        elif text[i] == '"':
                            # Handle quoted strings to avoid counting braces inside strings
                            i += 1
                            while i < len(text) and text[i] != '"':
                                if text[i] == '\\':
                                    i += 1  # Skip escaped character
                                i += 1
                        i += 1
                    
                    if brace_count == 0:
                        # Found complete JSON object
                        json_candidate = text[start:i]
                        json_objects.append(json_candidate)
                else:
                    i += 1
            
            return json_objects
        
        # Find all potential JSON objects
        json_objects = find_json_objects(text)
        
        for json_str in json_objects:
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                continue

        if not _repaired:
            repaired = repair_json_escapes(text)
            if repaired != text:
                return extract_json_from_text(repaired, json_key, _repaired=True)

    except Exception as e:
        print(f"Failed to extract JSON: {e}")
        if len(text) > 500:
            print(f"Raw content preview:\n{text[:500]}...")
        else:
            print(f"Raw content:\n{text}")
        
    return None

def extract_playbook_bullets(playbook_text, bullet_ids):
    """
    Extract specific bullet points from playbook based on bullet_ids.
    
    Args:
        playbook_text (str): The full playbook text
        bullet_ids (list): List of bullet IDs to extract
    
    Returns:
        str: Formatted playbook content containing only the specified bullets
    """
    if not bullet_ids:
        return "(No bullets used by generator)"
    
    lines = playbook_text.strip().split('\n')
    found_bullets = []
    
    for line in lines:
        if line.strip():  # Skip empty lines
            parsed = parse_playbook_line(line)
            if parsed and parsed['id'] in bullet_ids:
                found_bullets.append({
                    'id': parsed['id'],
                    'content': parsed['content'],
                    'helpful': parsed['helpful'],
                    'harmful': parsed['harmful']
                })
    
    if not found_bullets:
        return "(Generator referenced bullet IDs but none were found in playbook)"
    
    # Format the bullets for reflector input
    formatted_bullets = []
    for bullet in found_bullets:
        formatted_bullets.append(f"[{bullet['id']}] helpful={bullet['helpful']} harmful={bullet['harmful']} :: {bullet['content']}")
    
    return '\n'.join(formatted_bullets)
