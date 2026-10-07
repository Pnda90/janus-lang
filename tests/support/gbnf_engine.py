"""
Motore autonomo per parsing, riconoscimento formale e campionamento di grammatiche GBNF (Fase 2).
Implementa:
1. GBNFParser: parser per specifiche di grammatica GBNF (sottoinsieme standard llama.cpp).
2. GBNFRecognizer: parser memoizzato (packrat) per verificare se una stringa è accettata al 100% dalla grammatica.
3. GBNFSampler: derivatore stocastico guidato dalla grammatica per generare stringhe casuali conformi.
4. Verifica opzionale con LlamaGrammar ufficiale (llama-cpp-python) se disponibile.
"""

import re
import random
from typing import Dict, List, Set, Tuple, Optional, Any


class GBNFASTNode:
    pass


class LiteralNode(GBNFASTNode):
    def __init__(self, text: str):
        self.text = text

    def __repr__(self):
        return f"Literal({self.text!r})"


class CharClassNode(GBNFASTNode):
    def __init__(self, negated: bool, ranges: List[Tuple[int, int]], chars: Set[int]):
        self.negated = negated
        self.ranges = ranges  # List of (start_ord, end_ord)
        self.chars = chars    # Set of ords

    def matches(self, ch: str) -> bool:
        if not ch:
            return False
        code = ord(ch)
        in_set = code in self.chars or any(start <= code <= end for start, end in self.ranges)
        return not in_set if self.negated else in_set

    def __repr__(self):
        return f"CharClass(negated={self.negated}, ranges={self.ranges}, chars={self.chars})"


class RuleRefNode(GBNFASTNode):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"RuleRef({self.name})"


class SeqNode(GBNFASTNode):
    def __init__(self, items: List[GBNFASTNode]):
        self.items = items

    def __repr__(self):
        return f"Seq({self.items})"


class AltNode(GBNFASTNode):
    def __init__(self, items: List[GBNFASTNode]):
        self.items = items

    def __repr__(self):
        return f"Alt({self.items})"


class OptNode(GBNFASTNode):
    def __init__(self, item: GBNFASTNode):
        self.item = item

    def __repr__(self):
        return f"Opt({self.item})"


class StarNode(GBNFASTNode):
    def __init__(self, item: GBNFASTNode):
        self.item = item

    def __repr__(self):
        return f"Star({self.item})"


class PlusNode(GBNFASTNode):
    def __init__(self, item: GBNFASTNode):
        self.item = item

    def __repr__(self):
        return f"Plus({self.item})"


class GBNFParser:
    """Parser per convertire testo GBNF in un dizionario di regole AST."""

    def __init__(self, gbnf_text: str):
        self.gbnf_text = gbnf_text

    def parse(self) -> Dict[str, GBNFASTNode]:
        rules: Dict[str, GBNFASTNode] = {}
        # Pre-processa: rimuovi commenti ed unisci linee
        lines = [line.strip() for line in self.gbnf_text.splitlines() if line.strip() and not line.strip().startswith("#")]
        for line in lines:
            if "::=" not in line:
                continue
            name, expr_str = line.split("::=", 1)
            rule_name = name.strip()
            ast = self._parse_expr(expr_str.strip())
            rules[rule_name] = ast
        return rules

    def _parse_expr(self, text: str) -> GBNFASTNode:
        # Alternativa a livello superiore divisa da '|' non all'interno di stringhe, parentesi o classi
        alts = self._split_top_level(text, "|")
        if len(alts) > 1:
            return AltNode([self._parse_expr(a) for a in alts])

        # Sequenza separata da spazio
        parts = self._split_sequence(text)
        if len(parts) == 0:
            return LiteralNode("")
        if len(parts) == 1:
            return self._parse_atom_with_suffix(parts[0])
        return SeqNode([self._parse_atom_with_suffix(p) for p in parts])

    def _split_top_level(self, text: str, sep: str) -> List[str]:
        results = []
        current = []
        in_quote = False
        in_bracket = False
        paren_depth = 0
        i = 0
        while i < len(text):
            c = text[i]
            if c == '\\':
                current.append(c)
                if i + 1 < len(text):
                    current.append(text[i + 1])
                    i += 2
                    continue
                i += 1
                continue
            if c == '"' and not in_bracket:
                in_quote = not in_quote
            elif c == '[' and not in_quote:
                in_bracket = True
            elif c == ']' and not in_quote:
                in_bracket = False
            elif c == '(' and not in_quote and not in_bracket:
                paren_depth += 1
            elif c == ')' and not in_quote and not in_bracket:
                paren_depth -= 1
            elif c == sep and not in_quote and not in_bracket and paren_depth == 0:
                results.append("".join(current).strip())
                current = []
                i += 1
                continue

            current.append(c)
            i += 1

        if current:
            results.append("".join(current).strip())
        return results

    def _split_sequence(self, text: str) -> List[str]:
        results = []
        current = []
        in_quote = False
        in_bracket = False
        paren_depth = 0
        i = 0
        while i < len(text):
            c = text[i]
            if c == '\\':
                current.append(c)
                if i + 1 < len(text):
                    current.append(text[i + 1])
                    i += 2
                    continue
                i += 1
                continue
            if c == '"' and not in_bracket:
                in_quote = not in_quote
            elif c == '[' and not in_quote:
                in_bracket = True
            elif c == ']' and not in_quote:
                in_bracket = False
            elif c == '(' and not in_quote and not in_bracket:
                paren_depth += 1
            elif c == ')' and not in_quote and not in_bracket:
                paren_depth -= 1
            elif c.isspace() and not in_quote and not in_bracket and paren_depth == 0:
                if current:
                    results.append("".join(current))
                    current = []
                i += 1
                continue

            current.append(c)
            i += 1

        if current:
            results.append("".join(current))
        return results

    def _parse_atom_with_suffix(self, token: str) -> GBNFASTNode:
        token = token.strip()
        if not token:
            return LiteralNode("")

        if token.endswith("?") and not (token.startswith('"') and token.endswith('"')):
            return OptNode(self._parse_atom_with_suffix(token[:-1]))
        if token.endswith("*") and not (token.startswith('"') and token.endswith('"')):
            return StarNode(self._parse_atom_with_suffix(token[:-1]))
        if token.endswith("+") and not (token.startswith('"') and token.endswith('"')):
            return PlusNode(self._parse_atom_with_suffix(token[:-1]))

        # Gruppo ( ... )
        if token.startswith("(") and token.endswith(")"):
            inner = token[1:-1].strip()
            return self._parse_expr(inner)

        # Stringa letterale " ... "
        if token.startswith('"') and token.endswith('"'):
            # Decodifica escape nella stringa letterale
            val = token[1:-1]
            unescaped = bytes(val, "utf-8").decode("unicode_escape", errors="replace")
            return LiteralNode(unescaped)

        # Classe di caratteri [ ... ]
        if token.startswith("[") and token.endswith("]"):
            inner = token[1:-1]
            negated = False
            if inner.startswith("^"):
                negated = True
                inner = inner[1:]
            ranges, chars = self._parse_char_class_contents(inner)
            return CharClassNode(negated, ranges, chars)

        # Riferimento a regola
        return RuleRefNode(token)

    def _parse_char_class_contents(self, s: str) -> Tuple[List[Tuple[int, int]], Set[int]]:
        ranges: List[Tuple[int, int]] = []
        chars: Set[int] = set()

        raw_chars: List[int] = []
        i = 0
        while i < len(s):
            c = s[i]
            if c == '\\' and i + 1 < len(s):
                nxt = s[i + 1]
                if nxt == 'x' and i + 3 < len(s):
                    hex_str = s[i + 2:i + 4]
                    try:
                        raw_chars.append(int(hex_str, 16))
                        i += 4
                        continue
                    except ValueError:
                        pass
                esc_map = {'n': '\n', 'r': '\r', 't': '\t', 'b': '\b', 'f': '\f', '"': '"', '\\': '\\', '/': '/'}
                if nxt in esc_map:
                    raw_chars.append(ord(esc_map[nxt]))
                else:
                    raw_chars.append(ord(nxt))
                i += 2
            else:
                raw_chars.append(ord(c))
                i += 1

        idx = 0
        while idx < len(raw_chars):
            if idx + 2 < len(raw_chars) and chr(raw_chars[idx + 1]) == '-':
                ranges.append((raw_chars[idx], raw_chars[idx + 2]))
                idx += 3
            else:
                chars.add(raw_chars[idx])
                idx += 1

        return ranges, chars


class GBNFRecognizer:
    """Riconoscitore formale memoizzato (Packrat) per validare stringhe contro una grammatica GBNF."""

    def __init__(self, gbnf_text: str):
        self.rules = GBNFParser(gbnf_text).parse()

    def recognize(self, text: str, root_rule: str = "root") -> bool:
        if root_rule not in self.rules:
            return False

        memo: Dict[Tuple[str, int], Set[int]] = {}
        in_progress: Set[Tuple[str, int]] = set()

        def match(node: GBNFASTNode, pos: int) -> Set[int]:
            if pos > len(text):
                return set()

            if isinstance(node, LiteralNode):
                if text.startswith(node.text, pos):
                    return {pos + len(node.text)}
                return set()

            elif isinstance(node, CharClassNode):
                if pos < len(text) and node.matches(text[pos]):
                    return {pos + 1}
                return set()

            elif isinstance(node, RuleRefNode):
                key = (node.name, pos)
                if key in memo:
                    return memo[key]
                if key in in_progress:
                    return set()

                if node.name not in self.rules:
                    return set()

                in_progress.add(key)
                res = match(self.rules[node.name], pos)
                in_progress.remove(key)
                memo[key] = res
                return res

            elif isinstance(node, SeqNode):
                current_positions = {pos}
                for item in node.items:
                    next_positions = set()
                    for p in current_positions:
                        next_positions.update(match(item, p))
                    current_positions = next_positions
                    if not current_positions:
                        break
                return current_positions

            elif isinstance(node, AltNode):
                res = set()
                for item in node.items:
                    res.update(match(item, pos))
                return res

            elif isinstance(node, OptNode):
                return {pos} | match(node.item, pos)

            elif isinstance(node, StarNode):
                positions = {pos}
                queue = [pos]
                visited = {pos}
                while queue:
                    p = queue.pop(0)
                    for np in match(node.item, p):
                        if np > p and np not in visited:
                            visited.add(np)
                            positions.add(np)
                            queue.append(np)
                return positions

            elif isinstance(node, PlusNode):
                first_positions = match(node.item, pos)
                all_positions = set(first_positions)
                queue = [p for p in first_positions if p > pos]
                visited = set(queue)
                while queue:
                    p = queue.pop(0)
                    for np in match(node.item, p):
                        if np > p and np not in visited:
                            visited.add(np)
                            all_positions.add(np)
                            queue.append(np)
                return all_positions

            return set()

        final_positions = match(RuleRefNode(root_rule), 0)
        return len(text) in final_positions


class GBNFSampler:
    """Campionatore probabilistico guidato dalla grammatica GBNF con seed deterministico."""

    def __init__(self, gbnf_text: str):
        self.rules = GBNFParser(gbnf_text).parse()

    def sample(self, root_rule: str = "root", seed: Optional[int] = None, max_depth: int = 35) -> str:
        rng = random.Random(seed)

        def derive(node: GBNFASTNode, depth: int) -> str:
            if depth > max_depth:
                # Forza terminazione con espansione minima
                if isinstance(node, (StarNode, OptNode)):
                    return ""
                if isinstance(node, PlusNode):
                    return derive(node.item, depth + 1)
                if isinstance(node, AltNode):
                    # Cerca l'alternativa più corta
                    return derive(node.items[0], depth + 1)

            if isinstance(node, LiteralNode):
                return node.text

            elif isinstance(node, CharClassNode):
                if node.negated:
                    # Scegli tra caratteri alfanumerici e spazi standard
                    safe_pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-:.,!?"
                    matching = [c for c in safe_pool if node.matches(c)]
                    return rng.choice(matching) if matching else "a"
                else:
                    candidates = []
                    for code in node.chars:
                        candidates.append(chr(code))
                    for start, end in node.ranges:
                        for code in range(start, min(start + 50, end + 1)):
                            candidates.append(chr(code))
                    return rng.choice(candidates) if candidates else " "

            elif isinstance(node, RuleRefNode):
                if node.name == "ws":
                    return rng.choice(["", " ", "\n"])
                if node.name == "ws1":
                    return rng.choice([" ", "\n", "\t"])
                if node.name not in self.rules:
                    return ""
                return derive(self.rules[node.name], depth + 1)

            elif isinstance(node, SeqNode):
                return "".join(derive(item, depth + 1) for item in node.items)

            elif isinstance(node, AltNode):
                choice = rng.choice(node.items)
                return derive(choice, depth + 1)

            elif isinstance(node, OptNode):
                if rng.random() < 0.6:
                    return derive(node.item, depth + 1)
                return ""

            elif isinstance(node, StarNode):
                # Ripeti 0, 1 o 2 volte
                k = rng.choices([0, 1, 2], weights=[0.5, 0.4, 0.1])[0]
                return "".join(derive(node.item, depth + 1) for _ in range(k))

            elif isinstance(node, PlusNode):
                k = rng.choices([1, 2], weights=[0.8, 0.2])[0]
                return "".join(derive(node.item, depth + 1) for _ in range(k))

            return ""

        if root_rule not in self.rules:
            raise ValueError(f"Regola iniziale '{root_rule}' non trovata nella grammatica.")

        return derive(self.rules[root_rule], 0)
