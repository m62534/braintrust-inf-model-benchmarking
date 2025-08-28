#!/usr/bin/env python3
"""
Script to update all remaining test cases to use the new multiple expected answers structure.
"""

import re

def update_test_case(input_text, expected_text, expected_score):
    """Generate multiple expected answers for a test case."""
    
    # Create poor answer (factually incorrect or very vague)
    poor_answer = "This is not relevant to the question."
    
    # Create basic answer (minimal correct information)
    question_part = input_text.split("Question: ")[1].split("?")[0] if "Question: " in input_text else "The topic"
    basic_answer = f"{question_part} has multiple factors."
    
    # Create good answer (reasonable but not comprehensive)
    good_answer = expected_text.split(".")[0] + "."
    
    # Excellent answer is the original expected text
    excellent_answer = expected_text
    
    return {
        "poor": {
            "score": 0.0,
            "answer": poor_answer
        },
        "basic": {
            "score": 0.4,
            "answer": basic_answer
        },
        "good": {
            "score": 0.7,
            "answer": good_answer
        },
        "excellent": {
            "score": 1.0,
            "answer": excellent_answer
        }
    }

# Read the current dataset file
with open('src/dataset.py', 'r') as f:
    content = f.read()

# Find all test cases that still have the old structure
pattern = r'TestCase\(\s*input=\{\s*"input":\s*"([^"]+)",\s*"output":\s*"([^"]+)",\s*"expected":\s*"([^"]+)"\s*\},\s*expected=([0-9.]+),'
matches = re.findall(pattern, content, re.DOTALL)

print(f"Found {len(matches)} test cases to update")

# Update each test case
for i, (input_text, output_text, expected_text, expected_score) in enumerate(matches):
    print(f"Updating test case {i+1}: {input_text[:50]}...")
    
    # Generate new expected answers
    expected_answers = update_test_case(input_text, expected_text, float(expected_score))
    
    # Create the new test case structure
    new_structure = f'''TestCase(
        input={{
            "input": "{input_text}",
        }},
        expected_answers={{
            "poor": {{
                "score": {expected_answers["poor"]["score"]}, 
                "answer": "{expected_answers["poor"]["answer"]}"
            }},
            "basic": {{
                "score": {expected_answers["basic"]["score"]}, 
                "answer": "{expected_answers["basic"]["answer"]}"
            }},
            "good": {{
                "score": {expected_answers["good"]["score"]}, 
                "answer": "{expected_answers["good"]["answer"]}"
            }},
            "excellent": {{
                "score": {expected_answers["excellent"]["score"]}, 
                "answer": "{expected_answers["excellent"]["answer"]}"
            }}
        }},'''
    
    # Create a simpler pattern for replacement
    old_pattern = f'TestCase(\\s*input={{\\s*"input":\\s*"{re.escape(input_text)}",\\s*"output":\\s*"{re.escape(output_text)}",\\s*"expected":\\s*"{re.escape(expected_text)}"\\s*}},\\s*expected={expected_score},'
    content = re.sub(old_pattern, new_structure, content, flags=re.DOTALL)

# Write the updated content back
with open('src/dataset.py', 'w') as f:
    f.write(content)

print("✅ All test cases updated successfully!")
