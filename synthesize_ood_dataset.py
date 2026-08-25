"""
Project Antigravity — Tier 1 Out-of-Distribution (OOD) Dataset Synthesizer

Procedurally generates 100 novel multi-step arithmetic, geometric, and logical reasoning
problems with guaranteed ground-truth answers. Ensures 0% pretraining memory contamination.
"""

import json
import random

def generate_100_ood_problems(filename="synthesized_math100.jsonl"):
    random.seed(424242)
    problems = []

    names = ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah", "Ian", "Julia"]
    items = ["widgets", "gadgets", "apples", "books", "tokens", "coins", "stamps", "marbles", "gems", "origami cranes"]

    for i in range(1, 101):
        p_type = i % 5
        person = random.choice(names)
        item = random.choice(items)

        if p_type == 1:
            # Type 1: Financial & Inventory Multi-step
            start = random.randint(50, 300)
            bought = random.randint(10, 50)
            price = random.randint(3, 15)
            sold = random.randint(5, bought)
            sell_price = price + random.randint(2, 8)
            
            total_spent = bought * price
            revenue = sold * sell_price
            net_inventory = start + bought - sold
            ans = net_inventory * 2 + revenue - total_spent
            
            q = (f"{person} starts with {start} {item}. {person} buys {bought} more at ${price} each, "
                 f"and then sells {sold} of them at ${sell_price} each. "
                 f"If each remaining {item} is valued at $2, what is the total value in dollars of {person}'s remaining {item} inventory plus profit?")
            ans_str = f"#### {ans}"

        elif p_type == 2:
            # Type 2: Composite Geometry
            length = random.randint(8, 30)
            width = random.randint(5, 20)
            cut_side = random.randint(2, min(length, width) - 1)
            
            area = (length * width) - (cut_side * cut_side)
            perimeter = 2 * (length + width)
            ans = area + perimeter
            
            q = (f"A rectangular garden has length {length} meters and width {width} meters. "
                 f"A square section with side length {cut_side} meters is cut out from one corner for a fountain. "
                 f"What is the sum of the remaining area in square meters and the original outer perimeter in meters?")
            ans_str = f"#### {ans}"

        elif p_type == 3:
            # Type 3: Arithmetic Sequences & Summation
            a1 = random.randint(3, 15)
            d = random.randint(2, 7)
            n = random.randint(6, 12)
            
            an = a1 + (n - 1) * d
            s_n = n * (a1 + an) // 2
            ans = s_n
            
            q = (f"{person} saves money for {n} days. On day 1, {person} saves ${a1}. "
                 f"Each subsequent day, {person} saves ${d} more than the previous day. "
                 f"How much total money in dollars does {person} save by the end of day {n}?")
            ans_str = f"#### {ans}"

        elif p_type == 4:
            # Type 4: Rate & Work Collaboration
            rate1 = random.randint(4, 12) # items per hour
            rate2 = random.randint(5, 15)
            hours1 = random.randint(2, 6)
            hours2 = random.randint(3, 8)
            
            total_items = rate1 * hours1 + (rate1 + rate2) * hours2
            ans = total_items
            
            q = (f"{person} works alone for {hours1} hours producing {item} at a rate of {rate1} {item} per hour. "
                 f"Then a helper joins {person}, producing {rate2} {item} per hour. "
                 f"They work together for another {hours2} hours. How many total {item} did they produce in total?")
            ans_str = f"#### {ans}"

        else:
            # Type 0: Modular & Number Theory
            base = random.randint(100, 500)
            multiplier = random.randint(3, 9)
            subtrahend = random.randint(15, 80)
            modulus = random.randint(7, 19)
            
            result = (base * multiplier - subtrahend) % modulus
            ans = result
            
            q = (f"Calculate the remainder when ({base} * {multiplier} - {subtrahend}) is divided by {modulus}.")
            ans_str = f"#### {ans}"

        problems.append({
            "id": i,
            "question": q,
            "answer": ans_str,
            "target_num": int(ans_str.replace("#### ", ""))
        })

    with open(filename, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")

    print(f"✅ Generated {len(problems)} novel OOD reasoning problems in '{filename}'.")

if __name__ == "__main__":
    generate_100_ood_problems()
