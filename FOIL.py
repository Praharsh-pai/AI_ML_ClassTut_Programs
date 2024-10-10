family_tree = {
    'parent': [
        ('John', 'Mary'),    # John is the parent of Mary
        ('John', 'Paul'),    # John is the parent of Paul
        ('Alice', 'Mary'),   # Alice is the parent of Mary
        ('Alice', 'Paul'),   # Alice is the parent of Paul
        ('Mary', 'Jake'),    # Mary is the parent of Jake
        ('Paul', 'Lucy'),    # Paul is the parent of Lucy
    ]
}

def parent(x, y):
    return (x, y) in family_tree['parent']

def grandparent(x, y):
    for z in set(a for a, b in family_tree['parent']):
        if parent(x, z) and parent(z, y):
            return True
    return False

def foil_learn(target_relation, examples, facts):
    learned_rules = []
    for pos_example in examples['positive']:
        candidate_rule = []
        for fact_type, fact_data in facts.items():
            for fact in fact_data:
                if fact in examples['negative']:
                    continue
                candidate_rule.append(fact)

        for neg_example in examples['negative']:
            for fact_type, fact_data in facts.items():
                for fact in fact_data:
                    if fact in neg_example:
                        candidate_rule.remove(fact)
        learned_rules.append(candidate_rule)
    return learned_rules

examples = {
    'positive': [
        ('John', 'Jake'),  # John is a grandparent of Jake
        ('Alice', 'Jake')  # Alice is a grandparent of Jake
    ],
    'negative': [
        ('Paul', 'Lucy'),  # Paul is not a grandparent of Lucy
        ('Mary', 'Lucy')   # Mary is not a grandparent of Lucy
    ]
}

learned_rules = foil_learn(grandparent, examples, family_tree)

print("Learned Rules for Grandparent:")
for rule in learned_rules:
    print(rule)

print(f"Is John a grandparent of Jake? {grandparent('John', 'Jake')}")
print(f"Is Alice a grandparent of John? {grandparent('Alice', 'John')}")
print(f"Is Mary a grandparent of Lucy? {grandparent('Mary', 'Lucy')}")
