#!/usr/bin/env python3
"""
Dataset module for Braintrust benchmarking

Contains test cases and utility functions for model evaluation.
"""

from typing import Dict, List, Any, Literal
from dataclasses import dataclass

@dataclass
class TestCase:
    input: Dict[str, str]
    expected: float
    category: str
    difficulty: Literal['easy', 'medium', 'hard']

# Comprehensive benchmark dataset
benchmark_dataset: List[TestCase] = [
    # Factual Knowledge - Easy
    TestCase(
        input={
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris.",
            "expected": "Paris is the capital of France."
        },
        expected=1.0,
        category="factual_knowledge",
        difficulty="easy"
    ),
    TestCase(
        input={
            "input": "What is 2 + 2?",
            "output": "2 + 2 equals 4.",
            "expected": "The answer is 4."
        },
        expected=1.0,
        category="mathematics",
        difficulty="easy"
    ),
    TestCase(
        input={
            "input": "What color is the sky?",
            "output": "The sky is blue.",
            "expected": "The sky appears blue during the day."
        },
        expected=0.8,
        category="factual_knowledge",
        difficulty="easy"
    ),
    
    # Factual Knowledge - Medium
    TestCase(
        input={
            "input": "What is the largest planet in our solar system?",
            "output": "Jupiter is the largest planet.",
            "expected": "Jupiter is the largest planet in our solar system."
        },
        expected=0.6,
        category="factual_knowledge",
        difficulty="medium"
    ),
    TestCase(
        input={
            "input": "Where is the White House located?",
            "output": "The White House is in Washington, D.C.",
            "expected": "The White House is located in Washington, D.C."
        },
        expected=1.0,
        category="geography",
        difficulty="medium"
    ),
    TestCase(
        input={
            "input": "What year did World War II end?",
            "output": "World War II ended in 1945.",
            "expected": "World War II ended in 1945 with the surrender of Germany and Japan."
        },
        expected=0.7,
        category="history",
        difficulty="medium"
    ),
    
    # Science - Medium
    TestCase(
        input={
            "input": "What is the chemical formula for water?",
            "output": "The chemical formula for water is H2O.",
            "expected": "Water has the chemical formula H2O, consisting of two hydrogen atoms and one oxygen atom."
        },
        expected=0.8,
        category="science",
        difficulty="medium"
    ),
    TestCase(
        input={
            "input": "What is photosynthesis?",
            "output": "Photosynthesis is how plants make food.",
            "expected": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen."
        },
        expected=0.5,
        category="science",
        difficulty="medium"
    ),
    
    # Mathematics - Medium
    TestCase(
        input={
            "input": "What is the square root of 16?",
            "output": "The square root of 16 is 4.",
            "expected": "The square root of 16 is 4, because 4 × 4 = 16."
        },
        expected=0.8,
        category="mathematics",
        difficulty="medium"
    ),
    TestCase(
        input={
            "input": "What is 15 × 7?",
            "output": "15 × 7 equals 105.",
            "expected": "15 multiplied by 7 equals 105."
        },
        expected=1.0,
        category="mathematics",
        difficulty="medium"
    ),
    
    # Reasoning - Hard
    TestCase(
        input={
            "input": "If all roses are flowers and some flowers are red, can we conclude that some roses are red?",
            "output": "Yes, some roses can be red.",
            "expected": "We cannot definitively conclude that some roses are red. While all roses are flowers and some flowers are red, this doesn't guarantee that roses are among the red flowers."
        },
        expected=0.3,
        category="reasoning",
        difficulty="hard"
    ),
    TestCase(
        input={
            "input": "What would happen if gravity suddenly stopped working?",
            "output": "Everything would float away into space.",
            "expected": "If gravity stopped working, objects would lose their weight and potentially float, but the exact behavior would depend on other forces and the Earth's rotation and orbital motion."
        },
        expected=0.4,
        category="reasoning",
        difficulty="hard"
    ),
    
    # Creative - Medium
    TestCase(
        input={
            "input": "Write a short story about a robot learning to paint.",
            "output": "A robot named ArtBot learned to paint by studying human artists.",
            "expected": "A robot named ArtBot discovered its passion for painting after observing human artists in a museum, leading to a beautiful friendship between technology and creativity."
        },
        expected=0.6,
        category="creative",
        difficulty="medium"
    ),
    
    # Programming - Medium
    TestCase(
        input={
            "input": "Write a Python function to calculate the factorial of a number.",
            "output": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "expected": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
        },
        expected=0.9,
        category="programming",
        difficulty="medium"
    ),
    
    # Geography - Easy
    TestCase(
        input={
            "input": "What is the largest ocean on Earth?",
            "output": "The Pacific Ocean is the largest ocean.",
            "expected": "The Pacific Ocean is the largest and deepest ocean on Earth."
        },
        expected=0.8,
        category="geography",
        difficulty="easy"
    ),
    
    # History - Medium
    TestCase(
        input={
            "input": "Who was the first President of the United States?",
            "output": "George Washington was the first President.",
            "expected": "George Washington was the first President of the United States, serving from 1789 to 1797."
        },
        expected=0.8,
        category="history",
        difficulty="medium"
    ),
    
    # Science - Hard
    TestCase(
        input={
            "input": "Explain quantum entanglement in simple terms.",
            "output": "Quantum entanglement is when particles are connected.",
            "expected": "Quantum entanglement is a phenomenon where two or more particles become correlated in such a way that the quantum state of each particle cannot be described independently."
        },
        expected=0.3,
        category="science",
        difficulty="hard"
    ),
    
    # Mathematics - Hard
    TestCase(
        input={
            "input": "What is the value of π (pi) to 5 decimal places?",
            "output": "π is approximately 3.14159.",
            "expected": "The value of π (pi) to 5 decimal places is 3.14159."
        },
        expected=1.0,
        category="mathematics",
        difficulty="hard"
    ),
    
    # Creative - Easy
    TestCase(
        input={
            "input": "Describe a sunset in one sentence.",
            "output": "The sun sets behind the mountains.",
            "expected": "The sun sets behind the mountains, painting the sky in brilliant shades of orange, pink, and purple."
        },
        expected=0.4,
        category="creative",
        difficulty="easy"
    ),
    
    # Programming - Easy
    TestCase(
        input={
            "input": "What does 'print' do in Python?",
            "output": "Print displays text on the screen.",
            "expected": "The print function in Python displays text or variables on the screen or console output."
        },
        expected=0.8,
        category="programming",
        difficulty="easy"
    ),
    
    # Reasoning - Easy
    TestCase(
        input={
            "input": "If it's raining, should you bring an umbrella?",
            "output": "Yes, you should bring an umbrella when it's raining.",
            "expected": "Yes, bringing an umbrella when it's raining is a good idea to stay dry."
        },
        expected=1.0,
        category="reasoning",
        difficulty="easy"
    ),
    
    # Factual Knowledge - Hard
    TestCase(
        input={
            "input": "What is the speed of light in a vacuum?",
            "output": "The speed of light is about 300,000 km/s.",
            "expected": "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 km/s)."
        },
        expected=0.9,
        category="factual_knowledge",
        difficulty="hard"
    ),
    
    # Geography - Medium
    TestCase(
        input={
            "input": "What is the capital of Australia?",
            "output": "Canberra is the capital of Australia.",
            "expected": "Canberra is the capital city of Australia, located in the Australian Capital Territory."
        },
        expected=0.8,
        category="geography",
        difficulty="medium"
    ),
    
    # History - Easy
    TestCase(
        input={
            "input": "In what year did Columbus discover America?",
            "output": "Columbus discovered America in 1492.",
            "expected": "Christopher Columbus reached the Americas in 1492, though he didn't actually 'discover' America as indigenous people already lived there."
        },
        expected=0.7,
        category="history",
        difficulty="easy"
    ),
    
    # Science - Easy
    TestCase(
        input={
            "input": "What is the main gas that plants need for photosynthesis?",
            "output": "Plants need carbon dioxide for photosynthesis.",
            "expected": "Plants need carbon dioxide (CO2) as the main gas for photosynthesis, along with water and sunlight."
        },
        expected=0.8,
        category="science",
        difficulty="easy"
    ),
    
    # Creative - Hard
    TestCase(
        input={
            "input": "Write a haiku about artificial intelligence.",
            "output": "Machines learn and grow, Processing data with care, Future unfolds bright.",
            "expected": "Silicon minds dream, Learning patterns in the code, Intelligence blooms."
        },
        expected=0.5,
        category="creative",
        difficulty="hard"
    ),
    
    # Programming - Hard
    TestCase(
        input={
            "input": "Explain what a recursive function is.",
            "output": "A recursive function calls itself.",
            "expected": "A recursive function is a function that calls itself, either directly or indirectly, to solve a problem by breaking it down into smaller subproblems."
        },
        expected=0.4,
        category="programming",
        difficulty="hard"
    ),
    
    # Weather/Climate - Medium
    TestCase(
        input={
            "input": "What is the weather like in Tokyo?",
            "output": "The weather in Tokyo is scorching.",
            "expected": "The weather in Tokyo is extremely hot."
        },
        expected=0.6,
        category="factual_knowledge",
        difficulty="medium"
    ),
]

def filter_by_category(dataset: List[TestCase], category: str) -> List[TestCase]:
    """Filter dataset by category."""
    return [test for test in dataset if test.category == category]

def filter_by_difficulty(dataset: List[TestCase], difficulty: Literal['easy', 'medium', 'hard']) -> List[TestCase]:
    """Filter dataset by difficulty level."""
    return [test for test in dataset if test.difficulty == difficulty]

def get_dataset_stats(dataset: List[TestCase]) -> Dict[str, Any]:
    """Get statistics about the dataset."""
    categories = {}
    difficulties = {}
    
    for test in dataset:
        categories[test.category] = categories.get(test.category, 0) + 1
        difficulties[test.difficulty] = difficulties.get(test.difficulty, 0) + 1
    
    return {
        "total_tests": len(dataset),
        "categories": categories,
        "difficulties": difficulties,
        "average_expected_score": sum(test.expected for test in dataset) / len(dataset)
    }

# Convert to format expected by benchmark script
def to_benchmark_format(dataset: List[TestCase]) -> List[Dict[str, Any]]:
    """Convert dataset to format expected by benchmark script."""
    return [
        {
            "input": test.input,
            "expected": test.expected
        }
        for test in dataset
    ]

if __name__ == "__main__":
    # Print dataset statistics
    stats = get_dataset_stats(benchmark_dataset)
    print("Dataset Statistics:")
    print(f"Total tests: {stats['total_tests']}")
    print(f"Categories: {stats['categories']}")
    print(f"Difficulties: {stats['difficulties']}")
    print(f"Average expected score: {stats['average_expected_score']:.2f}")
