import itertools

family_facts = [
    ('parent', 'Alice', 'Bob'),
    ('parent', 'Alice', 'Charlie'),
    ('parent', 'David', 'Alice'),
    ('parent', 'Eve', 'Alice'),
    ('parent', 'Bob', 'Emily'),
    ('parent', 'Charlie', 'Frank')
]

def parent(X, Y):
    return ('parent', X, Y) in family_facts

def sibling(X, Y):
    return any(parent(P, X) and parent(P, Y) and X != Y for P, _, _ in family_facts)

def grandparent(X, Y):
    return any(parent(X, Z) and parent(Z, Y) for Z, _, _ in family_facts)

class FOIL:
    def __init__(self, facts, target_predicate, predicate_name):
        self.facts = facts
        self.target_predicate = target_predicate
        self.predicate_name = predicate_name

    def learn_rules(self):
        rules = []
        people = set(person for _, person1, person2 in self.facts for person in (person1, person2))
        for X, Y in itertools.permutations(people, 2):
            if self.target_predicate(X, Y):
                rule = self.construct_rule(X, Y)
                if rule:
                    rules.append(rule)
        return rules

    def construct_rule(self, X, Y):
        if self.predicate_name == 'sibling' and sibling(X, Y):
            return f"sibling({X}, {Y}) :- parent(P, {X}), parent(P, {Y})."
        elif self.predicate_name == 'grandparent' and grandparent(X, Y):
            return f"grandparent({X}, {Y}) :- parent({X}, Z), parent(Z, {Y})."
        return None

foil_sibling = FOIL(family_facts, sibling, 'sibling')
sibling_rules = foil_sibling.learn_rules()

foil_grandparent = FOIL(family_facts, grandparent, 'grandparent')
grandparent_rules = foil_grandparent.learn_rules()

print("Learned sibling rules:")
for rule in sibling_rules:
    print(rule)

print("\nLearned grandparent rules:")
for rule in grandparent_rules:
    print(rule)
