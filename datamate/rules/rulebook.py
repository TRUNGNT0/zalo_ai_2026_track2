"""Conservative readiness checks, not a substitute for query-specific validation."""

import re

REQUIREMENTS = {
    'revenue': ({'order_id', 'order_date', 'order_status', 'line_amount'}, {'rulebook.1', 'rulebook.2', 'rulebook.8'}),
    'receivables': ({'order_id', 'order_date', 'order_status', 'payment_method', 'order_payable', 'payment_date', 'payment_amount'}, {'rulebook.2', 'rulebook.4'}),
    'inventory_as_of': ({'product_id', 'movement_time', 'stock_balance'}, {'rulebook.2', 'rulebook.5'}),
    'gross_profit': ({'order_id', 'order_date', 'order_status', 'line_amount', 'quantity', 'unit_cost'}, {'rulebook.1', 'rulebook.5'}),
}


class Rulebook:
    """Parse business rules and assess catalog capability gaps."""

    @staticmethod
    def assess_capabilities(catalog):
        documented = {c['semantic_name'] for t in catalog['tables'] for c in t['columns'] if c['semantic_name']}
        rules = set(catalog.get('rule_refs', []))
        capabilities = {}
        for name, (columns, required_rules) in REQUIREMENTS.items():
            missing = sorted(columns - documented)
            missing_rules = sorted(required_rules - rules)
            # Presence alone does not establish compatible grains, status decoding,
            # return handling, temporal boundaries, or safe joins for a query.
            capabilities[name] = {'status': 'blocked' if missing or missing_rules else 'partial',
                                  'missing_semantics': missing, 'missing_rules': missing_rules,
                                  'required_review': ['grain_and_join_path', 'status_and_return_policy', 'time_and_unit_basis']}
        total = sum(len(t['columns']) for t in catalog['tables'])
        mapped = sum(c['semantic_name'] is not None for t in catalog['tables'] for c in t['columns'])
        return {'coverage': {'source_columns': total, 'documented_columns': mapped,
                             'unresolved_columns': total - mapped}, 'capabilities': capabilities}

    @staticmethod
    def parse(text: str) -> list[dict[str, str]]:
        sections = []
        current_id = 'rulebook.preamble'
        lines = []
        for line in text.splitlines():
            match = re.match(r'^##\s+(\d+)\.', line)
            if match:
                if '\n'.join(lines).strip():
                    sections.append({'evidence_id': current_id, 'text': '\n'.join(lines).strip()})
                current_id = f'rulebook.{match[1]}'
                lines = []
            lines.append(line)
        if '\n'.join(lines).strip():
            sections.append({'evidence_id': current_id, 'text': '\n'.join(lines).strip()})
        return sections
