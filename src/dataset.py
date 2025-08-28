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
    expected_answers: Dict[str, Dict[str, Any]]  # Multiple expected answers with scores
    category: str
    difficulty: Literal['easy', 'medium', 'hard']

# Comprehensive benchmark dataset focused on context understanding and inference
benchmark_dataset: List[TestCase] = [
    # Context Understanding - Easy
    TestCase(
        input={
            "input": "Context: John is a doctor who works at City Hospital. He has been working there for 10 years and specializes in cardiology. Question: What type of medical procedures is John most likely to perform?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "John performs brain surgeries and dental procedures."
            },
            "basic": {
                "score": 0.4, 
                "answer": "John performs medical procedures."
            },
            "good": {
                "score": 0.7, 
                "answer": "John would perform heart-related procedures since he's a cardiologist."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "John would perform cardiovascular procedures such as heart surgeries, cardiac catheterizations, and heart disease treatments, since he specializes in cardiology."
            }
        },
        category="context_understanding",
        difficulty="easy"
    ),
    
    # Inference - Easy
    TestCase(
        input={
            "input": "Context: Sarah left her house at 8:00 AM and arrived at work at 8:45 AM. Her office is 15 miles away. Question: What can we infer about Sarah's average travel speed?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "Sarah traveled very slowly and was late to work."
            },
            "basic": {
                "score": 0.4, 
                "answer": "Sarah traveled 15 miles in some time."
            },
            "good": {
                "score": 0.7, 
                "answer": "Sarah traveled 15 miles in 45 minutes, so her speed was about 20 mph."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "Sarah traveled 15 miles in 45 minutes, which means her average speed was 20 miles per hour (15 miles ÷ 0.75 hours = 20 mph)."
            }
        },
        category="inference",
        difficulty="easy"
    ),
    
    # Problem Context - Medium
    TestCase(
        input={
            "input": "Context: A company's revenue increased by 25% this year, but their profit margin decreased by 10%. Their operating costs went up by 30%. Question: What likely caused the profit margin to decrease despite revenue growth?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "The company made less money this year."
            },
            "basic": {
                "score": 0.4, 
                "answer": "The profit margin decreased because costs went up."
            },
            "good": {
                "score": 0.7, 
                "answer": "The profit margin decreased because operating costs increased more than revenue."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "The profit margin decreased because operating costs increased by 30%, which was higher than the 25% revenue increase, meaning costs grew faster than income."
            }
        },
        category="problem_context",
        difficulty="medium"
    ),
    
    # Multi-step Reasoning - Medium
    TestCase(
        input={
            "input": "Context: If all students who study hard get good grades, and all students who get good grades are eligible for scholarships, and Maria is a student who studies hard, what can we conclude about Maria?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "Maria is a student."
            },
            "basic": {
                "score": 0.4, 
                "answer": "Maria studies hard."
            },
            "good": {
                "score": 0.7, 
                "answer": "Maria is eligible for scholarships because she studies hard and gets good grades."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "Maria is eligible for scholarships. This follows from: 1) Maria studies hard, 2) Students who study hard get good grades, 3) Students with good grades are eligible for scholarships, therefore Maria is eligible."
            }
        },
        category="multi_step_reasoning",
        difficulty="medium"
    ),
    
    # Context Ambiguity - Medium
    TestCase(
        input={
            "input": "Context: 'The bank is closed.' Question: What are the possible meanings of this statement?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "The bank is not open."
            },
            "basic": {
                "score": 0.4, 
                "answer": "The bank could be closed for the day or permanently closed."
            },
            "good": {
                "score": 0.7, 
                "answer": "The bank could be closed for the day, permanently closed, or closed for renovations."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "This statement has multiple possible meanings: 1) The bank is closed for the day (temporarily), 2) The bank is permanently closed, 3) The river bank is closed due to flooding, 4) The bank is closed for renovations."
            }
        },
        category="context_understanding",
        difficulty="medium"
    ),
    
    # Implicit Information - Medium
    TestCase(
        input={
            "input": "Context: A restaurant owner notices that sales are down 20% this month compared to last month. The weather has been unusually cold and rainy. Question: What might be the connection between these observations?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "The restaurant is not doing well."
            },
            "basic": {
                "score": 0.4, 
                "answer": "The cold weather might be keeping people from going out to eat."
            },
            "good": {
                "score": 0.7, 
                "answer": "The cold and rainy weather likely reduced customer traffic to the restaurant."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "The cold and rainy weather likely reduced customer traffic to the restaurant, as people tend to stay home during bad weather rather than going out to eat, explaining the 20% sales decline."
            }
        },
        category="inference",
        difficulty="medium"
    ),
    
    # Complex Problem Context - Hard
    TestCase(
        input={
            "input": "Context: A software company releases a new app. In the first week, they get 10,000 downloads but only 500 active users. In week 2, they get 8,000 downloads and 1,200 active users. In week 3, they get 6,000 downloads but 2,000 active users. Question: What trend can be observed and what might it indicate?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "The app is not doing well."
            },
            "basic": {
                "score": 0.4, 
                "answer": "Downloads are decreasing but active users are increasing."
            },
            "good": {
                "score": 0.7, 
                "answer": "Downloads are decreasing but active users are increasing, suggesting the app is gaining loyal users."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "The trend shows decreasing downloads but increasing active users. This suggests the app is gaining a more engaged user base - fewer people are downloading it, but those who do are more likely to become active users, indicating improved user retention and satisfaction."
            }
        },
        category="problem_context",
        difficulty="hard"
    ),
    
    # Advanced Inference - Hard
    TestCase(
        input={
            "input": "Context: A study finds that people who drink 3 cups of coffee per day have a 15% lower risk of developing Type 2 diabetes. The study controlled for age, weight, and exercise habits. Question: What can we reasonably conclude from this study?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "Coffee prevents diabetes."
            },
            "basic": {
                "score": 0.4, 
                "answer": "There's a correlation between coffee consumption and lower diabetes risk."
            },
            "good": {
                "score": 0.7, 
                "answer": "There's a correlation between coffee consumption and lower diabetes risk, but we can't say coffee causes the reduction."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "We can conclude there's a correlation between moderate coffee consumption and reduced diabetes risk, but we cannot establish causation. The study shows an association, but other factors not controlled for (like diet, genetics, or lifestyle differences) might explain the relationship."
            }
        },
        category="inference",
        difficulty="hard"
    ),
    
    # Context-Dependent Problem Solving - Hard
    TestCase(
        input={
            "input": "Context: A city has traffic congestion during rush hours. The mayor proposes building more roads. An urban planner suggests improving public transportation instead. Question: What factors should be considered to evaluate these proposals?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "Building roads is better than public transportation."
            },
            "basic": {
                "score": 0.4, 
                "answer": "Factors include cost, environmental impact, and effectiveness."
            },
            "good": {
                "score": 0.7, 
                "answer": "Factors include cost, environmental impact, long-term effectiveness, and whether more roads would actually reduce congestion."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "Key factors to consider: 1) Induced demand (more roads often lead to more traffic), 2) Environmental impact and sustainability, 3) Cost-effectiveness and maintenance, 4) Equity and accessibility, 5) Long-term urban planning goals, 6) Public transportation's potential to reduce car dependency."
            }
        },
        category="problem_context",
        difficulty="hard"
    ),
    
    # Multi-step Logical Reasoning - Hard
    TestCase(
        input={
            "input": "Context: If A implies B, and B implies C, and we know that C is false, what can we conclude about A?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "A is true."
            },
            "basic": {
                "score": 0.4, 
                "answer": "If C is false, then B must be false."
            },
            "good": {
                "score": 0.7, 
                "answer": "If C is false, then B must be false, and if B is false, then A must be false."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "We can conclude that A is false. This follows from: 1) If A implies B, then if A is true, B must be true, 2) If B implies C, then if B is true, C must be true, 3) Since C is false, B must be false (contrapositive), 4) Since B is false, A must be false (contrapositive)."
            }
        },
        category="multi_step_reasoning",
        difficulty="hard"
    ),
    
    # Context Understanding - Easy
    TestCase(
        input={
            "input": "Context: A teacher notices that students who sit in the front row tend to participate more in class discussions. Question: What are possible explanations for this observation?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "Students in front participate more."
            },
            "basic": {
                "score": 0.4, 
                "answer": "Students in front might be more engaged or the teacher might call on them more."
            },
            "good": {
                "score": 0.7, 
                "answer": "Students in front might be more engaged, the teacher might call on them more, or they feel more accountable."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "Possible explanations: 1) Students who choose front seats are already more engaged, 2) Being closer to the teacher increases attention and participation, 3) The teacher might unconsciously call on front-row students more, 4) Front-row students feel more accountable and visible."
            }
        },
        category="context_understanding",
        difficulty="easy"
    ),
    
    # Inference from Data - Medium
    TestCase(
        input={
            "input": "Context: A survey shows that 80% of people who exercise regularly report feeling happier than those who don't exercise. Question: What can we infer from this data?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "Exercise makes people happy."
            },
            "basic": {
                "score": 0.4, 
                "answer": "There's a correlation between exercise and happiness."
            },
            "good": {
                "score": 0.7, 
                "answer": "There's a correlation between exercise and happiness, but we can't determine cause and effect."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "We can infer there's a correlation between regular exercise and self-reported happiness, but we cannot determine causation. It's possible that exercise causes happiness, that happier people are more likely to exercise, or that other factors influence both exercise habits and happiness levels."
            }
        },
        category="inference",
        difficulty="medium"
    ),
    
    # Problem Context Analysis - Medium
    TestCase(
        input={
            "input": "Context: A company's customer service department receives 50% more complaints in December than in other months. Question: What might explain this seasonal increase?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "The company is doing poorly in December."
            },
            "basic": {
                "score": 0.4, 
                "answer": "December has holidays and increased shopping."
            },
            "good": {
                "score": 0.7, 
                "answer": "December has holidays and increased shopping, which could lead to more customer interactions and complaints."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "Possible explanations: 1) Holiday shopping season increases customer volume and interactions, 2) Holiday stress affects both customers and staff, 3) Seasonal products or services may have different quality issues, 4) Staff shortages during holidays may reduce service quality, 5) Higher expectations during gift-giving season."
            }
        },
        category="problem_context",
        difficulty="medium"
    ),
    
    # Context-Dependent Decision Making - Hard
    TestCase(
        input={
            "input": "Context: A small business owner must choose between raising prices by 10% or reducing staff by 20% to maintain profitability. Question: What factors should influence this decision?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "Raising prices is always better."
            },
            "basic": {
                "score": 0.4, 
                "answer": "Factors include customer price sensitivity and staff importance."
            },
            "good": {
                "score": 0.7, 
                "answer": "Factors include customer price sensitivity, staff importance to quality, market competition, and long-term business impact."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "Key factors: 1) Customer price sensitivity and potential loss of customers, 2) Impact on service quality and customer satisfaction, 3) Market competition and pricing strategies, 4) Employee morale and retention, 5) Long-term business sustainability, 6) Current market conditions and economic climate, 7) Alternative cost-cutting measures."
            }
        },
        category="problem_context",
        difficulty="hard"
    ),
    
    # Complex Inference Chain - Hard
    TestCase(
        input={
            "input": "Context: A study finds that people who read fiction books have better empathy scores than those who don't. Another study shows that people with higher empathy are more likely to volunteer. Question: What might this suggest about the relationship between reading fiction and volunteering?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "Reading fiction causes people to volunteer."
            },
            "basic": {
                "score": 0.4, 
                "answer": "There's a relationship between reading fiction and volunteering."
            },
            "good": {
                "score": 0.7, 
                "answer": "Reading fiction might indirectly increase volunteering through improved empathy."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "This suggests a potential indirect relationship where reading fiction may increase empathy, which in turn increases the likelihood of volunteering. However, this is a correlation chain, not causation - other factors might influence both reading habits and volunteering behavior."
            }
        },
        category="inference",
        difficulty="hard"
    ),
    
    # Context Understanding - Easy
    TestCase(
        input={
            "input": "Context: A restaurant changes its menu from traditional Italian to fusion cuisine. Sales increase by 30% in the first month. Question: What might explain this increase?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "The restaurant is doing well."
            },
            "basic": {
                "score": 0.4, 
                "answer": "The new menu attracted new customers or increased interest from existing customers."
            },
            "good": {
                "score": 0.7, 
                "answer": "The new menu attracted new customers, existing customers were excited to try something new, or the change generated positive publicity."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "Possible explanations: 1) The new menu attracted new customers interested in fusion cuisine, 2) Existing customers were excited to try something new, 3) The change generated positive publicity and word-of-mouth, 4) The new menu items had higher profit margins, 5) Seasonal factors or other external changes coincided with the menu change."
            }
        },
        category="context_understanding",
        difficulty="easy"
    ),
    
    # Multi-step Problem Solving - Medium
    TestCase(
        input={
            "input": "Context: A student wants to improve their grades. They currently study 2 hours per day and get B grades. They learn that students who study 4 hours per day typically get A grades. Question: What should the student consider before increasing study time?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "The student should study more."
            },
            "basic": {
                "score": 0.4, 
                "answer": "The student should consider if they have time available."
            },
            "good": {
                "score": 0.7, 
                "answer": "The student should consider if they have 2 extra hours available and if other factors affect their grades."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "The student should consider: 1) Whether they have 2 additional hours available daily, 2) If study quality matters more than quantity, 3) Other factors affecting grades (sleep, stress, teaching quality), 4) Diminishing returns on study time, 5) Balance with other important activities, 6) Whether the correlation applies to their specific situation."
            }
        },
        category="multi_step_reasoning",
        difficulty="medium"
    ),
    
    # Context-Dependent Analysis - Hard
    TestCase(
        input={
            "input": "Context: A city experiences a 40% increase in bicycle accidents after installing bike lanes. Question: What are possible interpretations of this data?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "Bike lanes are dangerous."
            },
            "basic": {
                "score": 0.4, 
                "answer": "More people might be cycling due to the bike lanes."
            },
            "good": {
                "score": 0.7, 
                "answer": "More people might be cycling due to the bike lanes, or the lanes might be poorly designed."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "Possible interpretations: 1) More people are cycling due to the new lanes (increased exposure), 2) The bike lanes are poorly designed or implemented, 3) Drivers aren't accustomed to sharing the road with cyclists, 4) The lanes create new conflict points, 5) Better reporting of accidents, 6) The lanes attract less experienced cyclists, 7) The absolute number of accidents increased but the rate per cyclist might have decreased."
            }
        },
        category="problem_context",
        difficulty="hard"
    ),
    
    # Inference from Limited Information - Medium
    TestCase(
        input={
            "input": "Context: A company's social media posts get 50% more engagement on weekends than weekdays. Question: What might this tell us about their audience?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "Their audience is more active on weekends."
            },
            "basic": {
                "score": 0.4, 
                "answer": "Their audience might have more free time on weekends."
            },
            "good": {
                "score": 0.7, 
                "answer": "Their audience might have more free time on weekends to engage with social media."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "This suggests their audience has more free time on weekends, possibly indicating they're working professionals who are busy during weekdays, or that weekend content is more appealing, or that people generally have more leisure time for social media on weekends."
            }
        },
        category="inference",
        difficulty="medium"
    ),
    
    # Complex Context Understanding - Hard
    TestCase(
        input={
            "input": "Context: A study finds that people who work from home report higher job satisfaction but also higher stress levels. Question: What might explain this apparent contradiction?",
        },
        expected_answers={
            "poor": {
                "score": 0.0, 
                "answer": "Working from home is both good and bad."
            },
            "basic": {
                "score": 0.4, 
                "answer": "Working from home might have both positive and negative aspects."
            },
            "good": {
                "score": 0.7, 
                "answer": "Working from home might have both positive and negative aspects that affect satisfaction and stress differently."
            },
            "excellent": {
                "score": 1.0, 
                "answer": "This contradiction can be explained by: 1) Different aspects of work-from-home affect satisfaction vs. stress, 2) Higher satisfaction from flexibility but stress from blurred work-life boundaries, 3) Satisfaction from avoiding commute but stress from isolation, 4) Satisfaction from autonomy but stress from self-management, 5) The relationship between satisfaction and stress is complex and not always inverse."
            }
        },
        category="context_understanding",
        difficulty="hard"
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
        "average_expected_score": sum(test.expected_answers["excellent"]["score"] for test in dataset) / len(dataset)
    }

# Convert to format expected by benchmark script
def to_benchmark_format(dataset: List[TestCase]) -> List[Dict[str, Any]]:
    """Convert dataset to format expected by benchmark script."""
    return [
        {
            "input": test.input,
            "expected_answers": test.expected_answers
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
