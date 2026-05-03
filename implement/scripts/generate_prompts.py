"""
Generate training prompts for SUCA.

Creates compositional prompts with multiple semantic constraints:
  - Multi-object scenes
  - Attribute bindings (color, material, size)
  - Counting constraints
  - Spatial relations
"""

import json
import random
from pathlib import Path

OBJECTS = [
    "cat", "dog", "bird", "rabbit", "horse", "elephant", "giraffe",
    "butterfly", "fish", "turtle", "apple", "pear", "banana", "orange",
    "flower", "tree", "chair", "table", "car", "bicycle", "house",
    "book", "cup", "ball", "hat", "shoe", "clock", "lamp", "box",
    "bottle", "cake", "candle", "umbrella", "bag", "phone",
]

COLORS = [
    "red", "blue", "green", "yellow", "black", "white", "orange",
    "purple", "pink", "brown", "golden", "silver",
]

MATERIALS = [
    "wooden", "glass", "metal", "plastic", "stone", "paper",
    "ceramic", "crystal", "leather", "silk",
]

SIZES = ["tiny", "small", "large", "huge", "tall", "short"]

RELATIONS = [
    "on top of", "next to", "behind", "in front of", "inside",
    "above", "below", "beside", "between", "under",
]

BACKGROUNDS = [
    "on a wooden table", "in a garden", "on a beach",
    "in a living room", "on a mountain", "in a forest",
    "in a kitchen", "on a city street", "in a field of grass",
    "by a river", "in a snowy landscape", "in a library",
]

NUMBERS = ["two", "three", "four", "five"]


def generate_entity_attribute_prompt() -> str:
    """Generate: [color] [object] and [color] [object] [background]."""
    obj1, obj2 = random.sample(OBJECTS, 2)
    c1, c2 = random.sample(COLORS, 2)
    bg = random.choice(BACKGROUNDS)
    return f"a {c1} {obj1} and a {c2} {obj2} {bg}"


def generate_counting_prompt() -> str:
    """Generate: [number] [color] [objects] [background]."""
    obj = random.choice(OBJECTS) + "s"
    n = random.choice(NUMBERS)
    c = random.choice(COLORS)
    bg = random.choice(BACKGROUNDS)
    return f"{n} {c} {obj} {bg}"


def generate_spatial_prompt() -> str:
    """Generate: [obj1] [relation] [obj2] [background]."""
    obj1, obj2 = random.sample(OBJECTS, 2)
    c1, c2 = random.sample(COLORS, 2)
    rel = random.choice(RELATIONS)
    bg = random.choice(BACKGROUNDS)
    return f"a {c1} {obj1} {rel} a {c2} {obj2} {bg}"


def generate_complex_prompt() -> str:
    """Generate multi-constraint: [num] [color] [obj] and [num] [color] [obj] [rel] [obj]."""
    obj1, obj2, obj3 = random.sample(OBJECTS, 3)
    c1, c2, c3 = random.sample(COLORS, 3)
    n1, n2 = random.sample(NUMBERS, 2)
    rel = random.choice(RELATIONS)
    return f"{n1} {c1} {obj1}s and {n2} {c2} {obj2}s {rel} a {c3} {obj3}"


def generate_material_prompt() -> str:
    """Generate: [material] [object] with [color] [object]."""
    obj1, obj2 = random.sample(OBJECTS, 2)
    mat = random.choice(MATERIALS)
    c = random.choice(COLORS)
    bg = random.choice(BACKGROUNDS)
    return f"a {mat} {obj1} with a {c} {obj2} {bg}"


def generate_prompts(n: int = 1000, seed: int = 42) -> list[str]:
    random.seed(seed)
    generators = [
        generate_entity_attribute_prompt,
        generate_counting_prompt,
        generate_spatial_prompt,
        generate_complex_prompt,
        generate_material_prompt,
    ]

    prompts = set()
    while len(prompts) < n:
        gen = random.choice(generators)
        prompts.add(gen())

    return sorted(prompts)


if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "prompts.json"

    prompts = generate_prompts(1000)
    with open(out_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Generated {len(prompts)} prompts → {out_path}")
